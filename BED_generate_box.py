import random
import json
import copy
import os
from dataclass import Box, Pallet
from sort import RandomPacker
from basepacker import Packer, BinPacker
from utils import compute_metrics
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
import wandb
import multiprocessing
import numpy as np

def create_dim_converters(step: int, bit_length: int, min_dim: int):
    """
    编码规则：
    编码值 idx 对应尺寸 = idx * step + min_dim
    idx 从0开始，编码0对应最小尺寸 min_dim
    """
    def dim_to_bin_str(dim: int) -> str:
        idx = (dim - min_dim) // step
        if idx < 0 or idx >= 2 ** bit_length:
            raise ValueError(f"idx={idx} 超出 bit_length={bit_length} 的表示范围")
        return format(idx, f'0{bit_length}b')

    def bin_str_to_dim(bin_str: str) -> int:
        idx = int(bin_str, 2)
        return idx * step + min_dim

    return dim_to_bin_str, bin_str_to_dim

def fitness_wrapper(args):
    group_dims, pallet_dim, eval_times, min_lw, max_lw, min_h, max_h, step_lw, step_h, bit_length_lw, bit_length_h = args
    group = [Box(l, w, h) for l, w, h in group_dims]
    pallet = Pallet(*pallet_dim)
    score = 0

    k = 1  # 取最高1位进行熵计算，必要时可调
    weight_lw = 0.5  # 长宽熵奖励权重
    weight_h = 0.5    # 高度熵奖励权重

    # 定义编码函数（长宽取最大，编码离散等级）
    def encode_dim_lw(box):
        return (max(box.l, box.w) - min_lw) // step_lw

    def encode_dim_h(box):
        return (box.h - min_h) // step_h

    # 计算长宽熵奖励
    encoded_lw = [encode_dim_lw(box) for box in group]
    topkbits_lw = [(dim >> (bit_length_lw - k)) for dim in encoded_lw]
    counts_lw = np.bincount(topkbits_lw, minlength=2**k)
    probs_lw = counts_lw / np.sum(counts_lw) if np.sum(counts_lw) > 0 else np.array([1])
    probs_lw = probs_lw[probs_lw > 0]
    entropy_lw = -np.sum(probs_lw * np.log2(probs_lw)) if probs_lw.size > 0 else 0
    max_entropy_lw = np.log2(2**k)
    entropy_reward_lw = weight_lw * (entropy_lw / max_entropy_lw if max_entropy_lw > 0 else 0)

    # 计算高度熵奖励
    encoded_h = [encode_dim_h(box) for box in group]
    topkbits_h = [(dim >> (bit_length_h - k)) for dim in encoded_h]
    counts_h = np.bincount(topkbits_h, minlength=2**k)
    probs_h = counts_h / np.sum(counts_h) if np.sum(counts_h) > 0 else np.array([1])
    probs_h = probs_h[probs_h > 0]
    entropy_h = -np.sum(probs_h * np.log2(probs_h)) if probs_h.size > 0 else 0
    max_entropy_h = np.log2(2**k)
    entropy_reward_h = weight_h * (entropy_h / max_entropy_h if max_entropy_h > 0 else 0)

    entropy_reward = entropy_reward_lw + entropy_reward_h  # 总熵奖励

    # 统计长宽最大值和最小值均相等的箱子对数，作为惩罚依据
    lw_swap_pairs = 0
    group_len = len(group)
    for i in range(group_len):
        for j in range(i + 1, group_len):
            box_i = group[i]
            box_j = group[j]

            max_i, min_i = max(box_i.l, box_i.w), min(box_i.l, box_i.w)
            max_j, min_j = max(box_j.l, box_j.w), min(box_j.l, box_j.w)

            if max_i == max_j and min_i == min_j:
                lw_swap_pairs += 1

    penalty_weight_swap = 0.2  # 惩罚权重，可调节
    penalty_swap = penalty_weight_swap * lw_swap_pairs

    # 计算装箱利用率及未放箱惩罚
    for _ in range(eval_times):
        packer = RandomPacker(pallet, Packer)
        used_box = group.copy()
        random.shuffle(used_box)

        placed, unplaced = packer.pack(used_box)
        score -= len(unplaced)  # 未放箱子数量惩罚
        _, used_util, _ = compute_metrics(pallet, placed)
        score += float(used_util)  # 利用率奖励

    avg_score = score / eval_times
    final_score = avg_score + entropy_reward - penalty_swap

    return max(final_score, 0), -entropy_reward + penalty_swap




class BoxGeneratorGA:
    def __init__(self, config):
        self.best_score = None
        self.best_individual = None
        self.config = config

        self.num_boxes = config['num_boxes']
        self.pop_size = config['pop_size']
        self.generations = config['generations']
        self.mutation_rate = config['mutation_rate']
        self.elite_size = config.get('elite_size', 20)
        self.export_threshold = config.get('export_threshold', 0.7)
        self.eval_times = config.get('eval_times', 10)

        self.bit_length_lw = config.get('bit_length_lw', 6)
        self.step_lw = config.get('step_lw', 10)
        self.min_lw = config.get('min_lw', self.step_lw)
        self.max_lw = config.get('max_lw', self.step_lw * (2 ** self.bit_length_lw - 1) + self.min_lw)

        self.bit_length_h = config.get('bit_length_h', 5)
        self.step_h = config.get('step_h', 20)
        self.min_h = config.get('min_h', self.step_h)
        self.max_h = config.get('max_h', self.step_h * (2 ** self.bit_length_h - 1) + self.min_h)

        self.dim_to_bin_str_lw, self.bin_str_to_dim_lw = create_dim_converters(self.step_lw, self.bit_length_lw, self.min_lw)
        self.dim_to_bin_str_h, self.bin_str_to_dim_h = create_dim_converters(self.step_h, self.bit_length_h, self.min_h)

        self.name = config.get('name', 'box_generator')

        self.valid_lw_dims = [d for d in range(self.min_lw, self.max_lw + 1, self.step_lw)]
        self.valid_h_dims = [d for d in range(self.min_h, self.max_h + 1, self.step_h)]

        self.valid_dim_pairs = [
            (l, w, h)
            for l in self.valid_lw_dims
            for w in self.valid_lw_dims
            for h in self.valid_h_dims
            if h <= max(l, w)
        ]

        if not self.valid_dim_pairs:
            raise ValueError("⚠️ No valid box dimensions under constraints")

        self.population = self.init_population()
        self.pallet = Pallet(*config['pallet'])

        self.CR = config.get('CR', 0.7)  # 交叉概率

    def generate_box(self):
        while True:
            l, w, h = random.choice(self.valid_dim_pairs)
            if (
                self.min_lw <= l <= self.max_lw and
                self.min_lw <= w <= self.max_lw and
                self.min_h <= h <= self.max_h and
                h <= l and h <= w
            ):
                return Box(l, w, h)

    def init_population(self):
        return [[self.generate_box() for _ in range(self.num_boxes)] for _ in range(self.pop_size)]

    def crossover(self, target, population):
        CR = self.CR
        indices = list(range(len(population)))
        indices.remove(population.index(target))
        a, b, c = random.sample(indices, 3)
        a_group = population[a]
        b_group = population[b]
        c_group = population[c]

        child = []
        for i in range(self.num_boxes):
            new_dims = []
            for dim_key in ['l', 'w', 'h']:
                t_dim = getattr(target[i], dim_key)
                a_dim = getattr(a_group[i], dim_key)
                b_dim = getattr(b_group[i], dim_key)
                c_dim = getattr(c_group[i], dim_key)

                try:
                    if dim_key in ['l', 'w']:
                        t_bin = list(self.dim_to_bin_str_lw(t_dim))
                        a_bin = list(self.dim_to_bin_str_lw(a_dim))
                        b_bin = list(self.dim_to_bin_str_lw(b_dim))
                        c_bin = list(self.dim_to_bin_str_lw(c_dim))
                    else:
                        t_bin = list(self.dim_to_bin_str_h(t_dim))
                        a_bin = list(self.dim_to_bin_str_h(a_dim))
                        b_bin = list(self.dim_to_bin_str_h(b_dim))
                        c_bin = list(self.dim_to_bin_str_h(c_dim))
                except ValueError:
                    new_dims.append(t_dim)
                    continue

                diff_bin = [str(int(bi) ^ int(ci)) for bi, ci in zip(b_bin, c_bin)]
                v_bin = [str(int(ai) ^ int(di)) for ai, di in zip(a_bin, diff_bin)]
                trial_bin = [vi if random.random() < CR else ti for vi, ti in zip(v_bin, t_bin)]

                if dim_key in ['l', 'w']:
                    new_dim = self.bin_str_to_dim_lw("".join(trial_bin))
                    new_dim = max(self.min_lw, min(new_dim, self.max_lw))
                else:
                    new_dim = self.bin_str_to_dim_h("".join(trial_bin))
                    new_dim = max(self.min_h, min(new_dim, self.max_h))

                new_dims.append(new_dim)

            l, w, h = new_dims
            if not (h <= l and h <= w):
                h = min(l, w)
            child.append(Box(l, w, h))
        return child

    def mutate_box(self, box, max_attempts=10):
        for _ in range(max_attempts):
            choice = random.choice(['l', 'w', 'h'])
            dim = getattr(box, choice)
            try:
                if choice in ['l', 'w']:
                    bin_str = self.dim_to_bin_str_lw(dim)
                    bit_len = self.bit_length_lw
                else:
                    bin_str = self.dim_to_bin_str_h(dim)
                    bit_len = self.bit_length_h
            except ValueError:
                continue
            bit_idx = random.randint(0, bit_len - 1)
            flipped = list(bin_str)
            flipped[bit_idx] = '1' if flipped[bit_idx] == '0' else '0'

            if choice in ['l', 'w']:
                new_dim = self.bin_str_to_dim_lw(''.join(flipped))
                new_dim = max(self.min_lw, min(new_dim, self.max_lw))
            else:
                new_dim = self.bin_str_to_dim_h(''.join(flipped))
                new_dim = max(self.min_h, min(new_dim, self.max_h))

            if choice == 'l':
                l, w, h = new_dim, box.w, box.h
            elif choice == 'w':
                l, w, h = box.l, new_dim, box.h
            else:
                l, w, h = box.l, box.w, new_dim

            if not (h <= l and h <= w):
                h = min(l, w)

            return Box(l, w, h)
        return box

    def mutate(self, group):
        return [self.mutate_box(box) if random.random() < self.mutation_rate else box for box in group]

    def evolve(self):
        best_score = float("-inf")
        pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
        max_workers = min(10, multiprocessing.cpu_count())
        for gen in trange(self.generations, desc="进化中", ncols=80):
            args_list = [
                ([(b.l, b.w, b.h) for b in group],
                 pallet_dim, self.eval_times,
                 self.min_lw, self.max_lw,
                 self.min_h, self.max_h,
                 self.step_lw, self.step_h,
                 self.bit_length_lw, self.bit_length_h)
                for group in self.population
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                fitness_scores = list(executor.map(fitness_wrapper, args_list))
            scores, penalties = zip(*fitness_scores)
            scored_population = list(zip(scores, penalties, self.population))
            scored_population.sort(key=lambda x: x[0], reverse=True)
            self.population = [ind for _, _, ind in scored_population]
            current_best = self.population[0]
            current_score = scored_population[0][0]
            current_penalty = scored_population[0][1]
            wandb.log({"generation": gen,
                       "current_true_score": current_score + current_penalty,
                       "current_penalty": current_penalty,
                       "current_score": current_score})
            if current_score > best_score and current_score > self.export_threshold:
                best_score = current_score
                self.best_individual = copy.deepcopy(current_best)

                filepath = f"./{self.num_boxes}/{gen}bestBoxes{best_score:.4f}.json"
                self.export_best_to_json(filepath=filepath)
            if (gen + 1) % 50 == 0:
                print(f"Generation {gen + 1}, Best Fitness: {current_score:.4f}")
            next_gen = self.population[:self.elite_size]
            num_random = max(1, self.pop_size // 10)
            num_offspring = self.pop_size - self.elite_size - num_random
            for _ in range(num_offspring):
                target = random.choice(self.population[:self.elite_size])
                child = self.crossover(target, self.population)
                child = self.mutate(child)
                next_gen.append(child)
            for _ in range(num_random):
                new_ind = [self.generate_box() for _ in range(self.num_boxes)]
                next_gen.append(new_ind)
            self.population = next_gen
        self.best_score = best_score
        wandb.log({"final_best_score": best_score})


    def print_best_boxes(self):
        boxes = self.best_individual
        if boxes is None:
            return
        print("\nBest Boxes:")
        for i, box in enumerate(boxes):
            print(f"{i + 1:02d}: {box.l} \u00d7 {box.w} \u00d7 {box.h} mm")

    def export_best_to_json(self, filepath="best_boxes.json", save_config=True):
        boxes = self.best_individual
        if boxes is None:
            return
        box_list = [{"l": box.l, "w": box.w, "h": box.h} for box in boxes]
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(filepath, "w") as f:
            json.dump(box_list, f, indent=2)
        print(f"✅ 导出成功：{filepath}")

        # 同路径保存config.json，若文件已存在则不覆盖
        if self.config is not None and save_config:
            config_path = os.path.join(dir_path, "config.json")
            if not os.path.exists(config_path):
                with open(config_path, "w", encoding="utf-8") as cf:
                    json.dump(self.config, cf, indent=2, ensure_ascii=False)
                print(f"✅ 配置文件已保存：{config_path}")


if __name__ == "__main__":
    box_nums = [2,5,8,11,14,17,20,23]
    # box_nums = [2]
    for num in box_nums:
        dir = str(num)
        config = None
        for filename in os.listdir(dir):
            if filename =='config.json':
                full_path = os.path.join(dir, filename)
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    config = data
                    # print(data)
        config['pop_size']=10
        config["elite_size"] = int(0.1 * config["pop_size"])
        name = (
            f"{config['num_boxes']}boxes_"
            f"lwBL{config['bit_length_lw']}s{config['step_lw']}_"
            f"hBL{config['bit_length_h']}s{config['step_h']}"
        )
        config["name"] = name
        wandb.init(
            project=f"generate_boxes",
            name=name,
            config=config
        )

        ga = BoxGeneratorGA(config)
        ga.evolve()
        ga.print_best_boxes()
        wandb.finish()

        from group_test import BatchBoxTester


        pallet = Pallet(*config['pallet'])
        folder = f"./{config['num_boxes']}"  # 请替换为你的json目录

        tester = BatchBoxTester(pallet, folder, rounds=1000, is_random=False)
        results = tester.batch_test()
        tester.save_to_csv()









    # config = {
    #     "num_boxes": 2,
    #     "pallet": (1600, 1000, 1800),
    #     "pop_size": 50,
    #     "generations": 100,
    #     "mutation_rate": 0.15,
    #     "export_threshold": 0.1,
    #     "CR": 0.1,
    #     "eval_times": 100,
    #     "bit_length_lw": 6,
    #     "step_lw": 10,
    #     "min_lw": 500,
    #     "max_lw": 1000,
    #     "bit_length_h": 6,
    #     "step_h": 10,
    #     "min_h": 500,
    #     "max_h": 1000
    # }
    # config["elite_size"] = int(0.1 * config["pop_size"])
    # name = (
    #     f"lwBL{config['bit_length_lw']}s{config['step_lw']}_"
    #     f"hBL{config['bit_length_h']}s{config['step_h']}"
    # )
    #
    # config["name"] = name
    # wandb.init(
    #     project=f"generate_boxes",
    #     name=name,
    #     config=config
    # )
    #
    # ga = BoxGeneratorGA(config)
    # ga.evolve()
    # ga.print_best_boxes()
    # wandb.finish()
    #
    # from group_test import BatchBoxTester
    #
    #
    # pallet = Pallet(*config['pallet'])
    # folder = f"./{config['num_boxes']}"  # 请替换为你的json目录
    #
    # tester = BatchBoxTester(pallet, folder, rounds=1000, is_random=False)
    # results = tester.batch_test()
    # tester.save_to_csv()


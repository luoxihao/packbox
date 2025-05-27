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

def create_dim_converters(min_dim: int, step: int, bit_length: int):
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
    group_dims, pallet_dim, eval_times, min_dim, step, bit_length = args
    group = [Box(l, w, h) for l, w, h in group_dims]
    pallet = Pallet(*pallet_dim)
    score = 0

    # 参数，最高k位参与熵计算，建议1~3
    k = 1
    weight = 0.15  # 奖励权重，调节熵的影响力度

    # 先提取长宽最大值对应的编码（0~2^bit_length-1）
    def encode_dim(d):
        return (d - min_dim) // step

    max_lw_dims = []
    for box in group:
        l_enc = encode_dim(box.l)
        w_enc = encode_dim(box.w)
        max_lw_dims.append(max(l_enc, w_enc))

    # 提取最高k位
    topkbits = [ (dim >> (bit_length - k)) for dim in max_lw_dims ]

    # 计算熵
    counts = np.bincount(topkbits, minlength=2**k)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs)) if probs.size > 0 else 0

    max_entropy = np.log2(2**k)
    entropy_reward = weight * (entropy / max_entropy)

    # 包装利用率打分
    for _ in range(eval_times):
        packer = RandomPacker(pallet, Packer)
        used_box = group.copy()
        random.shuffle(used_box)

        placed, _ = packer.pack(used_box)
        _, used_util, _ = compute_metrics(pallet, placed)
        score += float(used_util)

    final_score = (score / eval_times) + entropy_reward
    return max(final_score, 0), -entropy_reward  # 返回负值作为penalty保持接口一致



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
        self.bit_length = config.get('bit_length', 7)
        self.min_dim = config.get('min_dim', 150)
        self.max_dim = config.get('max_dim', 1000)
        self.step = config.get('step', 10)
        self.CR = config.get('CR', 0.7)
        self.dim_to_bin_str, self.bin_str_to_dim = create_dim_converters(self.min_dim, self.step, self.bit_length)
        self.name = (
            f"numBoxes{self.num_boxes}_pop{self.pop_size}_"
            f"mut{self.mutation_rate}_"
            f"bitLen{self.bit_length}_minDim{self.min_dim}_"
            f"maxDim{self.max_dim}_step{self.generations}"
        )
        self.valid_dims = [
            d for d in (self.min_dim + i * self.step for i in range(2 ** self.bit_length))
            if self.min_dim <= d <= self.max_dim
        ]
        self.valid_dim_pairs = [
            (l, w, h) for l in self.valid_dims for w in self.valid_dims for h in self.valid_dims
            if h <= l and h <= w
        ]
        if not self.valid_dim_pairs:
            raise ValueError("⚠️ No valid box dimensions under constraints")
        self.population = self.init_population()
        self.pallet = Pallet(1600, 1000, 4000)

    def generate_box(self):
        while True:
            l, w, h = random.choice(self.valid_dim_pairs)
            if (
                self.min_dim <= l <= self.max_dim and
                self.min_dim <= w <= self.max_dim and
                self.min_dim <= h <= self.max_dim and
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
                    t_bin = list(self.dim_to_bin_str(t_dim))
                    a_bin = list(self.dim_to_bin_str(a_dim))
                    b_bin = list(self.dim_to_bin_str(b_dim))
                    c_bin = list(self.dim_to_bin_str(c_dim))
                except ValueError:
                    new_dims.append(t_dim)
                    continue

                diff_bin = [str(int(bi) ^ int(ci)) for bi, ci in zip(b_bin, c_bin)]
                v_bin = [str(int(ai) ^ int(di)) for ai, di in zip(a_bin, diff_bin)]
                trial_bin = [vi if random.random() < CR else ti for vi, ti in zip(v_bin, t_bin)]

                new_dim = self.bin_str_to_dim("".join(trial_bin))
                if self.min_dim <= new_dim <= self.max_dim:
                    new_dims.append(new_dim)
                else:
                    new_dims.append(t_dim)

            l, w, h = new_dims
            if not (h <= l and h <= w):
                h = min(l, w)  # 启发式修复
            child.append(Box(l, w, h))
        return child

    def mutate_box(self, box, max_attempts=10):
        for _ in range(max_attempts):
            choice = random.choice(['l', 'w', 'h'])
            dim = getattr(box, choice)
            try:
                bin_str = self.dim_to_bin_str(dim)
            except ValueError:
                continue
            bit_idx = random.randint(0, self.bit_length - 1)
            flipped = list(bin_str)
            flipped[bit_idx] = '1' if flipped[bit_idx] == '0' else '0'
            new_dim = self.bin_str_to_dim(''.join(flipped))
            if not (self.min_dim <= new_dim <= self.max_dim):
                continue

            if choice == 'l':
                l, w, h = new_dim, box.w, box.h
            elif choice == 'w':
                l, w, h = box.l, new_dim, box.h
            else:
                l, w, h = box.l, box.w, new_dim

            if not (h <= l and h <= w):
                h = min(l, w)  # 启发式修复

            return Box(l, w, h)
        return box

    def mutate(self, group):
        return [self.mutate_box(box) if random.random() < self.mutation_rate else box for box in group]

    def evolve(self):
        best_score = float("-inf")
        best_individual = None
        pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
        max_workers = min(10, multiprocessing.cpu_count())
        for gen in trange(self.generations, desc="进化中", ncols=80):
            args_list = [([(b.l, b.w, b.h) for b in group],
                        pallet_dim, self.eval_times, self.min_dim, self.step,self.bit_length) for group in self.population]

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
                    "current_true_score": current_score+current_penalty,
                    "current_penalty": current_penalty,
                    "current_score":current_score})
            if current_score > best_score and current_score > self.export_threshold:
                best_score = current_score
                self.best_individual = copy.deepcopy(current_best)
                # 生成name字符串（同wandb的命名规则）

                filepath = f"./{self.name}/{gen}bestBoxes{best_score:.4f}.json"
                self.export_best_to_json(filepath=filepath)
            if (gen + 1) % 50 == 0:
                print(f"Generation {gen + 1}, Best Fitness: {current_score:.4f}")
            next_gen = self.population[:self.elite_size]
            num_random = max(1, self.pop_size // 10)
            num_offspring = self.pop_size - self.elite_size - num_random
            for _ in range(num_offspring):
                target = random.choice(self.population[: self.elite_size])
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

    def export_best_to_json(self, filepath="best_boxes.json"):
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

if __name__ == "__main__":
    #mut 0.15 CR 0.085 box 10
    #mut 0.15 CR 0.1 box 25
    #mut 0.14 CR 0.07 box 50

    config = {
        "num_boxes": 25,
        "pop_size": 5000,
        "generations": 100,
        "mutation_rate": 0.15,
        "export_threshold": 0.1,
        "CR": 0.1,
        "eval_times": 100,
        "bit_length": 6,
        "min_dim": 150,
        "max_dim": 800,
        "step": 10
    }
    config["elite_size"] = int(0.1 * config["pop_size"])
    name = (
        f"debug_numBoxes{config['num_boxes']}_pop{config['pop_size']}_"
        f"mut{config['mutation_rate']}_CR{config['CR']}"
        f"bitLen{config['bit_length']}_minDim{config['min_dim']}_"
        f"maxDim{config['max_dim']}_step{config['step']}"
    )
    # name = (
    #     f"mut{config['mutation_rate']}_CR{config['CR']}"
    # )
    wandb.init(
        project=f"debug_{config['bit_length']}_eval{config['eval_times']}_step{config['step']}",
        name=name,
        config=config
    )

    ga = BoxGeneratorGA(config)
    ga.evolve()
    ga.print_best_boxes()
    wandb.finish()

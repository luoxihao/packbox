    import random
    import json
    import math
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
    from collections import defaultdict

    import numpy as np

    def fitness_wrapper(args):
        group_dims, pallet_dim, eval_times = args
        group = [Box(l, w, h) for l, w, h in group_dims]
        pallet = Pallet(*pallet_dim)
        score = 0
        penalty_sum = 0
        # 计算尺寸熵
        def compute_entropy(dim_values):
            # 将尺寸值离散化到计数器中
            counts = np.bincount(dim_values, minlength=64)  # 假设6位二进制，0-63
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]  # 避免log(0)
            entropy = -np.sum(probs * np.log2(probs)) if probs.size > 0 else 0
            return entropy

        # 提取所有箱子的l, w, h
        lengths = [box.l // 10 for box in group]  # 转换为0-63范围
        widths = [box.w // 10 for box in group]
        heights = [box.h // 10 for box in group]

        # 计算每个维度的熵
        entropy_l = compute_entropy(lengths)
        entropy_w = compute_entropy(widths)
        entropy_h = compute_entropy(heights)
        avg_entropy = (entropy_l + entropy_w + entropy_h) / 3

        # 最大熵（均匀分布，64个值）约为6（log2(64)）
        max_entropy = np.log2(64)  # 6.0
        target_entropy = max_entropy * 0.7  # 目标熵，70%最大熵
        entropy_penalty = 0.1 * abs(avg_entropy - target_entropy)  # 权重0.1，控制惩罚力度
        penalty_sum += entropy_penalty

        for _ in range(eval_times):
            packer = RandomPacker(pallet, Packer)
            binpacker = RandomPacker(pallet, BinPacker)
            used_box = random.choices(group, k=20)  # 修正为固定20个箱子

            placed, unplaced = packer.pack(used_box)
            util, used_util, _ = compute_metrics(pallet, placed)
            score += float(used_util)

            placed, unplaced = binpacker.pack(used_box)
            util, used_util, _ = compute_metrics(pallet, placed)
            score += float(used_util)

        # 归一化分数并减去惩罚
        final_score = (score / (eval_times * 2)) - penalty_sum
        return max(final_score, 0), penalty_sum  # 确保分数非负

    class BoxGeneratorGA:
        def __init__(self, config):
            self.config = config
            self.num_boxes = config['num_boxes']
            self.pop_size = config['pop_size']
            self.generations = config['generations']
            self.mutation_rate = config['mutation_rate']
            self.elite_size = config.get('elite_size', 20)
            self.export_threshold = config.get('export_threshold', 0.7)
            self.eval_times = config.get('eval_times', 10)
            self.bit_length = config.get('bit_length', 6)
            self.min_dim = config.get('min_dim', 10)
            self.max_dim = config.get('max_dim', 500)
            self.valid_dims = [
                d for d in (10 + i * 10 for i in range(2 ** self.bit_length))
                if self.min_dim <= d <= self.max_dim
            ]
            self.valid_dim_pairs = [
                (l, w, h) for l in self.valid_dims for w in self.valid_dims for h in self.valid_dims
                # if max(l, w, h) / min(l, w, h) <= 5
            ]
            if not self.valid_dim_pairs:
                raise ValueError("\u26a0\ufe0f No valid box dimensions under constraints")

            self.l_fixed = defaultdict(list)
            self.w_fixed = defaultdict(list)
            self.h_fixed = defaultdict(list)
            for l, w, h in self.valid_dim_pairs:
                self.l_fixed[(w, h)].append((l, w, h))
                self.w_fixed[(l, h)].append((l, w, h))
                self.h_fixed[(l, w)].append((l, w, h))

            self.population = self.init_population()
            self.pallet = Pallet(1200, 1000, 1800)

        def generate_box(self):
            l, w, h = random.choice(self.valid_dim_pairs)
            return Box(l, w, h)

        def init_population(self):
            return [[self.generate_box() for _ in range(self.num_boxes)] for _ in range(self.pop_size)]

        def crossover(self, parent1, parent2):
            point = self.num_boxes // 2
            return parent1[:point] + parent2[point:]

        def mutate_box(self, box):
            choice = random.choice(['l', 'w', 'h'])
            if choice == 'l':
                candidates = self.l_fixed.get((box.w, box.h), [])
            elif choice == 'w':
                candidates = self.w_fixed.get((box.l, box.h), [])
            else:
                candidates = self.h_fixed.get((box.l, box.w), [])
            if not candidates:
                return box
            return Box(*random.choice(candidates))

        def mutate(self, group):
            return [
                self.mutate_box(box) if random.random() < self.mutation_rate else box for box in group]

        def evolve(self):
            best_score = float("-inf")
            best_individual = None
            pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
            max_workers = min(10, multiprocessing.cpu_count())
            for gen in trange(self.generations, desc="\u8fdb\u5316\u4e2d", ncols=80):
                args_list = [([(b.l, b.w, b.h) for b in group], pallet_dim, self.eval_times) for group in self.population]
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    fitness_scores = list(executor.map(fitness_wrapper, args_list))
                scores, penalties = zip(*fitness_scores)
                scored_population = list(zip(scores, penalties, self.population))
                scored_population.sort(key=lambda x: x[0], reverse=True)
                self.population = [ind for _, _, ind in scored_population]
                current_best = self.population[0]
                current_score = scored_population[0][0]
                current_penalty = scored_population[0][1]
                wandb.log({"generation": gen, "current_score": current_score, "current_penalty": current_penalty})
                if current_score > best_score and current_score > self.export_threshold:
                    best_score = current_score
                    best_individual = copy.deepcopy(current_best)
                    self.export_best_to_json(
                        filepath=f"./binery{self.bit_length}_best_individual/best_boxes{best_score:.4f}.json")
                if (gen + 1) % 50 == 0:
                    print(f"Generation {gen + 1}, Best Fitness: {current_score:.4f}")
                next_gen = self.population[:self.elite_size]
                num_random = max(1, self.pop_size // 10)
                num_offspring = self.pop_size - self.elite_size - num_random
                for _ in range(num_offspring):
                    p1, p2 = random.choices(self.population[:max(10, self.elite_size * 5)], k=2)
                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    next_gen.append(child)
                for _ in range(num_random):
                    new_ind = [self.generate_box() for _ in range(self.num_boxes)]
                    next_gen.append(new_ind)
                self.population = next_gen
            self.best_individual = best_individual
            self.best_score = best_score
            wandb.log({"final_best_score": best_score})

        def get_best_boxes(self):
            pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
            max_workers = min(10, multiprocessing.cpu_count())
            args_list = [([(b.l, b.w, b.h) for b in group], pallet_dim, self.eval_times) for group in self.population]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                fitness_scores = list(executor.map(fitness_wrapper, args_list))
            scores, _ = zip(*fitness_scores)
            best_idx = scores.index(max(scores))
            return self.population[best_idx]

        def print_best_boxes(self):
            boxes = self.get_best_boxes()
            print("\nBest 25 Boxes:")
            for i, box in enumerate(boxes):
                print(f"{i + 1:02d}: {box.l} \u00d7 {box.w} \u00d7 {box.h} mm")

        def export_best_to_json(self, filepath="best_boxes.json"):
            boxes = self.get_best_boxes()
            box_list = [{"l": box.l, "w": box.w, "h": box.h} for box in boxes]
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(filepath, "w") as f:
                json.dump(box_list, f, indent=2)
            print(f"\u2705 \u5bfc\u51fa\u6210\u529f\uff1a{filepath}")

    if __name__ == "__main__":
        config = {
            "num_boxes": 25,
            "pop_size": 2000,
            "generations": 400,
            "mutation_rate": 0.2,
            "elite_size": 5,
            "export_threshold": 0.8,
            "eval_times": 5,
            "bit_length": 6,
            "min_dim": 100,
            "max_dim": 800
        }
        wandb.init(project="box_genetic_algorithm", name="GA_BinaryEncoded", config=config)
        ga = BoxGeneratorGA(config)
        ga.evolve()
        ga.print_best_boxes()
        ga.export_best_to_json("best_boxes.json")
        wandb.finish()
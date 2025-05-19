import random
import json
import math
import copy
import os
from dataclass import Box, Pallet
from sort import RandomPacker
from basepacker import Packer, EMSLBPacker, BinPacker
from utils import compute_metrics
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
import wandb
import multiprocessing


def binary_to_dim(binary_str):
    return int(binary_str, 2) * 10 + 10


def dim_to_binary(dim, bit_length):
    return format((dim - 10) // 10, f'0{bit_length}b')


def fitness_wrapper(args):
    group_dims, pallet_dim, eval_times = args
    group = [Box(l, w, h) for l, w, h in group_dims]  # 重建箱子，减少序列化开销
    pallet = Pallet(*pallet_dim)
    log_group_size = math.log(len(group))  # 提前计算，减少重复计算

    score = 0
    penalty_sum = 0
    for _ in range(eval_times):
        binpacker = RandomPacker(pallet, BinPacker)
        placed, unplaced = binpacker.pack(group)
        penalty = math.log(len(unplaced) + 1) / log_group_size
        penalty_sum += penalty
        util, used_util, _ = compute_metrics(pallet, placed)
        score += float(used_util)

    return score / eval_times, penalty_sum / eval_times


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

        # 预计算有效尺寸组合，避免拒绝采样
        self.valid_dims = [10 + i * 10 for i in range(2 ** self.bit_length)]
        self.valid_dim_pairs = [
            (l, w, h) for l in self.valid_dims for w in self.valid_dims for h in self.valid_dims
            if max(l, w, h) / min(l, w, h) <= 5
        ]

        self.population = self.init_population()
        self.pallet = Pallet(1200, 1000, 1800)
        self.fitness_cache = {}  # 适应度缓存：{group_tuple: (score, penalty)}

    def generate_box(self):
        # 直接从预计算的尺寸组合中采样
        l, w, h = random.choice(self.valid_dim_pairs)
        return Box(l, w, h)

    def init_population(self):
        return [
            [self.generate_box() for _ in range(self.num_boxes)]
            for _ in range(self.pop_size)
        ]

    def crossover(self, parent1, parent2):
        point = self.num_boxes // 2
        return parent1[:point] + parent2[point:]

    def mutate_box(self, box):
        if random.random() >= self.mutation_rate:
            return box
        choice = random.choice(['l', 'w', 'h'])

        # 找到有效变异，扰动一个维度
        valid_mutations = [
            (l, w, h) for l, w, h in self.valid_dim_pairs
            if (choice == 'l' and w == box.w and h == box.h) or
               (choice == 'w' and l == box.l and h == box.h) or
               (choice == 'h' and l == box.l and w == box.w)
        ]

        if not valid_mutations:
            return box
        new_l, new_w, new_h = random.choice(valid_mutations)
        return Box(new_l, new_w, new_h)

    def mutate(self, group):
        return [
            self.mutate_box(box) if random.random() < self.mutation_rate else box
            for box in group
        ]

    def evolve(self):
        best_score = float("-inf")
        best_individual = None
        pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
        max_workers = min(10, multiprocessing.cpu_count())

        for gen in trange(self.generations, desc="进化中", ncols=80):
            fitness_scores = []
            uncached_groups = []
            group_to_tuple = []

            # 收集需要计算的 group
            for group in self.population:
                group_tuple = tuple((b.l, b.w, b.h) for b in group)
                if group_tuple in self.fitness_cache:
                    fitness_scores.append(self.fitness_cache[group_tuple])
                else:
                    group_to_tuple.append(group_tuple)
                    uncached_groups.append(group)

            # 并行计算新 group 的适应度
            if uncached_groups:
                args_list = [([(b.l, b.w, b.h) for b in group], pallet_dim, self.eval_times) for group in
                             uncached_groups]
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    new_scores = list(executor.map(fitness_wrapper, args_list))
                for group_tuple, score in zip(group_to_tuple, new_scores):
                    self.fitness_cache[group_tuple] = score

            # 汇总所有 group 的 score（按 self.population 顺序）
            fitness_scores.clear()
            for group in self.population:
                group_tuple = tuple((b.l, b.w, b.h) for b in group)
                fitness_scores.append(self.fitness_cache[group_tuple])

            # 解包评分并排序
            scores, penalties = zip(*fitness_scores)
            scored_population = list(zip(scores, penalties, self.population))
            scored_population.sort(key=lambda x: x[0], reverse=True)

            self.population = [ind for _, _, ind in scored_population]
            current_best = self.population[0]
            current_score = scored_population[0][0]
            current_penalty = scored_population[0][1]

            # wandb 日志
            wandb.log({
                "generation": gen,
                "current_score": current_score,
                "current_penalty": current_penalty,
            })

            # 导出最佳个体
            if current_score > best_score and current_score > self.export_threshold:
                best_score = current_score
                best_individual = copy.deepcopy(current_best)
                self.export_best_to_json(
                    filepath=f"./binery{self.bit_length}_best_individual/best_boxes{best_score:.4f}.json")

            if (gen + 1) % 50 == 0:
                print(f"Generation {gen + 1}, Best Fitness: {current_score:.4f}")

            # 精英保留 + 交叉变异
            next_gen = self.population[:self.elite_size]
            while len(next_gen) < self.pop_size:
                p1, p2 = random.choices(self.population[:self.elite_size], k=2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
            self.population = next_gen

        self.best_individual = best_individual
        self.best_score = best_score
        wandb.log({"final_best_score": best_score})

    def get_best_boxes(self):
        pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)
        max_workers = min(10, multiprocessing.cpu_count())

        fitness_scores = []
        uncached_groups = []
        group_to_tuple = []

        # 筛选需要评估的 group
        for group in self.population:
            group_tuple = tuple((b.l, b.w, b.h) for b in group)
            if group_tuple in self.fitness_cache:
                fitness_scores.append(self.fitness_cache[group_tuple])
            else:
                group_to_tuple.append(group_tuple)
                uncached_groups.append(group)

        # 计算未缓存 group 的适应度
        if uncached_groups:
            args_list = [([(b.l, b.w, b.h) for b in group], pallet_dim, self.eval_times) for group in uncached_groups]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                new_scores = list(executor.map(fitness_wrapper, args_list))
            for group_tuple, score in zip(group_to_tuple, new_scores):
                self.fitness_cache[group_tuple] = score

        # 汇总所有适应度
        fitness_scores.clear()
        for group in self.population:
            group_tuple = tuple((b.l, b.w, b.h) for b in group)
            fitness_scores.append(self.fitness_cache[group_tuple])

        scores, _ = zip(*fitness_scores)
        best_idx = scores.index(max(scores))
        return self.population[best_idx]

    def print_best_boxes(self):
        boxes = self.get_best_boxes()
        print("\nBest 25 Boxes:")
        for i, box in enumerate(boxes):
            print(f"{i + 1:02d}: {box.l} × {box.w} × {box.h} mm")

    def export_best_to_json(self, filepath="best_boxes.json"):
        boxes = self.get_best_boxes()
        box_list = [{"l": box.l, "w": box.w, "h": box.h} for box in boxes]

        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(filepath, "w") as f:
            json.dump(box_list, f, indent=2)

        print(f"✅ 导出成功：{filepath}")


if __name__ == "__main__":
    config = {
        "num_boxes": 25,
        "pop_size": 20,
        "generations": 30000,
        "mutation_rate": 0.2,
        "elite_size": 2,
        "export_threshold": 0.8,
        "eval_times": 5,
        "bit_length": 7
    }

    wandb.init(project="box_genetic_algorithm", name="GA_BinaryEncoded", config=config)
    ga = BoxGeneratorGA(config)
    ga.evolve()
    ga.print_best_boxes()
    ga.export_best_to_json("best_boxes.json")
    wandb.finish()
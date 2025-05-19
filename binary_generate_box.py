import random
import json
import math
import copy
import os
from dataclass import Box, Pallet
from sort import RandomPacker
from basepacker import Packer, EMSLBPacker,BinPacker
from utils import compute_metrics
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
import wandb

def binary_to_dim(binary_str):
    return int(binary_str, 2) * 10 + 10

def dim_to_binary(dim, bit_length):
    return format((dim - 10) // 10, f'0{bit_length}b')

def fitness_wrapper(args):
    group_serialized, pallet_dim, eval_times = args
    group = [Box(**b) for b in group_serialized]
    pallet = Pallet(*pallet_dim)
    # packer = RandomPacker(pallet, Packer)
    # emslbpacker = RandomPacker(pallet, EMSLBPacker)
    
    score = 0
    for _ in range(eval_times):
        # placed, unplaced = packer.pack(group)
        # score -= math.log(len(unplaced) + 1)
        # util, used_util, _ = compute_metrics(pallet, placed)
        # score += util + used_util

        # placed, unplaced = emslbpacker.pack(group)
        # score -= math.log(len(unplaced) + 1)
        # util, used_util, _ = compute_metrics(pallet, placed)
        # score += util + used_util
        
        binpacker = RandomPacker(pallet, BinPacker)
        boxes_used = random.choices(group, k=random.randint(30, 40))
        placed, unplaced = binpacker.pack(boxes_used)
        penalty=math.log(len(unplaced) + 1)/(math.log(len(boxes_used)))
        score -= penalty
        util, used_util, _ = compute_metrics(pallet, placed)
        score +=  float(used_util)

    # return score / (eval_times * 2)
    return score / eval_times, penalty/eval_times

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

        self.population = self.init_population()
        self.pallet = Pallet(1200, 1000, 1800)

    def generate_box(self):
        while True:
            l_bin = format(random.randint(0, 2 ** self.bit_length - 1), f'0{self.bit_length}b')
            w_bin = format(random.randint(0, 2 ** self.bit_length - 1), f'0{self.bit_length}b')
            h_bin = format(random.randint(0, 2 ** self.bit_length - 1), f'0{self.bit_length}b')
            l, w, h = binary_to_dim(l_bin), binary_to_dim(w_bin), binary_to_dim(h_bin)
            if max(l, w, h) / min(l, w, h) <= 2:
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
        l_bin = dim_to_binary(box.l, self.bit_length)
        w_bin = dim_to_binary(box.w, self.bit_length)
        h_bin = dim_to_binary(box.h, self.bit_length)
        choice = random.choice(['l', 'w', 'h'])
        index = random.randint(0, self.bit_length - 1)

        def flip_bit(s, i):
            return s[:i] + ('1' if s[i] == '0' else '0') + s[i+1:]

        if choice == 'l':
            l_bin = flip_bit(l_bin, index)
        elif choice == 'w':
            w_bin = flip_bit(w_bin, index)
        else:
            h_bin = flip_bit(h_bin, index)

        l, w, h = binary_to_dim(l_bin), binary_to_dim(w_bin), binary_to_dim(h_bin)
        if max(l, w, h) / min(l, w, h) <= 2:
            return Box(l, w, h)
        return box

    def mutate(self, group):
        return [
            self.mutate_box(box) if random.random() < self.mutation_rate else box
            for box in group
        ]

    def evolve(self):
        best_score = float("-inf")
        best_individual = None
        pallet_dim = (self.pallet.l, self.pallet.w, self.pallet.h)

        for gen in trange(self.generations, desc="Evolving", ncols=80):
            args_list = [([b.__dict__ for b in group], pallet_dim, self.eval_times) for group in self.population]

            with ProcessPoolExecutor(max_workers=10) as executor:
                fitness_scores = list(executor.map(fitness_wrapper, args_list))

            # 拆分 score 和 penalty
            scores, penalties = zip(*fitness_scores)
            scores = list(scores)
            penalties = list(penalties)

            # 按 score 排序种群
            scored_population = list(zip(scores, penalties, self.population))
            scored_population.sort(key=lambda x: x[0], reverse=True)

            self.population = [ind for _, _, ind in scored_population]
            current_best = self.population[0]
            current_score = scored_population[0][0]
            current_penalty = scored_population[0][1]

            # 记录到 wandb
            wandb.log({
                "generation": gen,
                "current_score": current_score,
                "current_penalty": current_penalty,
            })

            if current_score > best_score and current_score > self.export_threshold:
                best_score = current_score
                best_individual = copy.deepcopy(current_best)
                self.export_best_to_json(filepath=f"./binery7_best_individual/best_boxes{best_score:.4f}.json")

            if (gen + 1) % 50 == 0:
                print(f"Generation {gen + 1}, Best Fitness: {current_score:.4f}")

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
        args_list = [([b.__dict__ for b in group], pallet_dim, self.eval_times) for group in self.population]
        with ProcessPoolExecutor(max_workers=10) as executor:
            fitness_scores = list(executor.map(fitness_wrapper, args_list))
        # 拆分为两个列表
        scores, penalties = zip(*fitness_scores)

        # 找到最大分数对应索引
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
        "pop_size": 100,
        "generations": 30000,
        "mutation_rate": 0.2,
        "elite_size": 10,
        "export_threshold": 0.8,
        "eval_times": 10,
        "bit_length": 7
    }

    wandb.init(project="box_genetic_algorithm", name="GA_BinaryEncoded", config=config)
    ga = BoxGeneratorGA(config)
    ga.evolve()
    ga.print_best_boxes()
    ga.export_best_to_json("best_boxes.json")
    wandb.finish()

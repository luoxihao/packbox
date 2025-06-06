import random
import json
import math
import copy
import os
from dataclass import Box, Pallet
from sort import RandomPacker
from basepacker import Packer, EMSLBPacker
from utils import compute_metrics
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
import wandb

def fitness_wrapper(args):
    group, pallet_dim, eval_times = args
    pallet = Pallet(*pallet_dim)
    packer = RandomPacker(pallet, Packer)
    emslbpacker = RandomPacker(pallet, EMSLBPacker)

    score = 0
    for _ in range(eval_times):
        placed, unplaced = packer.pack(group)
        score -= math.log(len(unplaced) + 1)
        util, used_util, _ = compute_metrics(pallet, placed)
        score += util + used_util

        placed, unplaced = emslbpacker.pack(group)
        score -= math.log(len(unplaced) + 1)
        util, used_util, _ = compute_metrics(pallet, placed)
        score += util + used_util

    return score / (eval_times * 2)

class BoxGeneratorGA:
    def __init__(self, config):
        self.config = config
        self.num_boxes = config['num_boxes']
        self.pop_size = config['pop_size']
        self.generations = config['generations']
        self.dim_limit = config['dim_limit']
        self.mutation_rate = config['mutation_rate']
        self.elite_size = config.get('elite_size', 20)
        self.export_threshold = config.get('export_threshold', 0.7)
        self.delta_ratio = config.get('delta_ratio', 0.5)
        self.eval_times = config.get('eval_times', 10)

        self.population = self.init_population()
        self.pallet = Pallet(1200, 1000, 1800)

    def generate_box(self):
        while True:
            l = random.randint(*self.dim_limit)
            w = random.randint(*self.dim_limit)
            h = random.randint(*self.dim_limit)
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
        def mutate_dim(dim):
            delta = int(dim * self.delta_ratio)
            delta = max(1, delta)
            new_dim = dim + random.randint(-delta, delta)
            return max(self.dim_limit[0], min(self.dim_limit[1], new_dim))

        for _ in range(10):
            l = mutate_dim(box.l)
            w = mutate_dim(box.w)
            h = mutate_dim(box.h)
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
            args_list = [(group, pallet_dim, self.eval_times) for group in self.population]

            with ProcessPoolExecutor() as executor:
                fitness_scores = list(executor.map(fitness_wrapper, args_list))

            scored_population = list(zip(fitness_scores, self.population))
            scored_population.sort(key=lambda x: x[0], reverse=True)
            self.population = [ind for _, ind in scored_population]
            current_best = self.population[0]
            current_score = scored_population[0][0]

            wandb.log({"generation": gen, "current_score": current_score})

            if current_score > best_score and current_score > self.export_threshold:
                best_score = current_score
                best_individual = copy.deepcopy(current_best)
                self.export_best_to_json(filepath=f"./best_individual/best_boxes{best_score:.4f}.json")

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
        args_list = [(group, pallet_dim, self.eval_times) for group in self.population]
        with ProcessPoolExecutor() as executor:
            fitness_scores = list(executor.map(fitness_wrapper, args_list))
        best_idx = fitness_scores.index(max(fitness_scores))
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
        "pop_size": 50,
        "generations": 30000,
        "dim_limit": (50, 1000),
        "mutation_rate": 0.2,
        "elite_size": 20,
        "export_threshold": 0.7,
        "delta_ratio": 0.5,
        "eval_times": 10
    }
    wandb.init(project="box_genetic_algorithm", name="GA_Run", config=config)

    ga = BoxGeneratorGA(config)
    ga.evolve()
    ga.print_best_boxes()
    ga.export_best_to_json("best_boxes.json")
    wandb.finish()

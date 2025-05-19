import random
from dataclass import Box,Pallet
import json
from sort import RandomPacker
from basepacker import Packer,EMSLBPacker
from utils import compute_metrics
import math
from tqdm import trange
import copy
class BoxGeneratorGA:
    def __init__(self, num_boxes=25, pop_size=10, generations=30,
                dim_limit=(50, 1000), mutation_rate=0.1):
        self.num_boxes = num_boxes              # 每组多少个箱子
        self.pop_size = pop_size                # 每代多少组
        self.generations = generations          # 演化多少代
        self.dim_limit = dim_limit              # 长宽高允许范围
        self.mutation_rate = mutation_rate      # 每个箱子被变异的概率
        self.population = self.init_population()

        self.pallet = Pallet(1200,1000,1800)
        self.packer = RandomPacker(self.pallet,Packer)
        self.emslbpacker = RandomPacker(self.pallet,Packer)

    def generate_box(self):
        while True:
            l = random.randint(*self.dim_limit)
            w = random.randint(*self.dim_limit)
            h = random.randint(*self.dim_limit)
            max_side = max(l, w, h)
            min_side = min(l, w, h)
            if max_side / min_side <= 2:
                return Box(l, w, h)

    def init_population(self):
        return [
            [self.generate_box() for _ in range(self.num_boxes)]
            for _ in range(self.pop_size)
        ]

    def fitness(self, group):
        score = 0
        for _ in range(10):
            placed, unplaced = self.packer.pack(group)
            score -= math.log(len(unplaced)+1)
            util, used_util, height = compute_metrics(self.pallet, placed)
            score += util + used_util

            placed, unplaced = self.emslbpacker.pack(group)
            score -= math.log(len(unplaced)+1)
            util, used_util, height = compute_metrics(self.pallet, placed)
            score += util + used_util

        return score / 10

    def crossover(self, parent1, parent2):
        point = self.num_boxes // 2
        return parent1[:point] + parent2[point:]

    def mutate_box(self, box, delta_ratio=0.01):
        def mutate_dim(dim):
            delta = int(dim * delta_ratio)
            delta = max(1, delta)
            new_dim = dim + random.randint(-delta, delta)
            return max(self.dim_limit[0], min(self.dim_limit[1], new_dim))

        for _ in range(10):  # 最多尝试10次防止死循环
            l = mutate_dim(box.l)
            w = mutate_dim(box.w)
            h = mutate_dim(box.h)
            max_side = max(l, w, h)
            min_side = min(l, w, h)
            if max_side / min_side <= 2:
                return Box(l, w, h)
        
        # 如果尝试多次仍不满足，则返回原 box（保守处理）
        return box

    def mutate(self, group):
        return [
            self.mutate_box(box) if random.random() < self.mutation_rate else box
            for box in group
        ]


    def evolve(self):
        best_score = float("-inf")
        best_individual = None

        for gen in trange(self.generations, desc="Evolving", ncols=80):
            self.population.sort(key=self.fitness, reverse=True)
            current_best = self.population[0]
            current_score = self.fitness(current_best)

            # 更新全局最佳
            if current_score > best_score:
                best_score = current_score
                best_individual = copy.deepcopy(current_best)

            # 每 N 代打印一次
            if (gen + 1) % 50 == 0:
                print(f"Generation {gen+1}, Best Fitness: {current_score:.4f}")

            # 进化新一代
            next_gen = self.population[:2]  # 精英保留
            while len(next_gen) < self.pop_size:
                p1, p2 = random.choices(self.population[:5], k=2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
            self.population = next_gen

        # 保存最终最佳个体（给类内成员）
        self.best_individual = best_individual
        self.best_score = best_score

        # 可选：自动导出
        self.export_individual_to_json(best_individual, "best_overall_boxes.json")
        print(f"\n✅ 最优个体保存完毕，最终 fitness = {best_score:.4f}")

    def get_best_boxes(self):
        best = max(self.population, key=self.fitness)
        return best

    def print_best_boxes(self):
        boxes = self.get_best_boxes()
        print("\nBest 25 Boxes:")
        for i, box in enumerate(boxes):
            print(f"{i+1:02d}: {box.l} × {box.w} × {box.h} mm")
    
    def export_best_to_json(self, filepath="best_boxes.json"):
        boxes = self.get_best_boxes()
        box_list = [
            {"l": box.l, "w": box.w, "h": box.h}
            for box in boxes
        ]
        with open(filepath, "w") as f:
            json.dump(box_list, f, indent=2)
        print(f"✅ 导出成功：{filepath}")


if __name__ == "__main__":
    # 实例化遗传算法
    ga = BoxGeneratorGA(
        num_boxes=25,
        pop_size=10,
        generations=3000,
        dim_limit=(50, 1000),
        mutation_rate=0.1,
    )

    # 运行进化过程
    ga.evolve()

    # 打印最优解
    ga.print_best_boxes()

    # 导出为 JSON 文件
    ga.export_best_to_json("best_boxes.json")
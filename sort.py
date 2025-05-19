
import random
from utils import check_available
from dataclass import Box, Pallet
import itertools
import math


class GreedyPacker:
    def __init__(self, pallet,packer):
        self.pallet = pallet
        self.packer  = packer(pallet)
    def pack(self, boxes):
        boxes = sorted(boxes, key=lambda b: b.l * b.w * b.h, reverse=True)
        return self.packer.pack(boxes)

class RandomPacker:
    def __init__(self, pallet,packer):
        self.pallet = pallet
        self.packer  = packer(pallet)
    def pack(self, boxes):
        random.shuffle(boxes)
        return self.packer.pack(boxes)

class SearchPacker:
    def __init__(self, pallet, packer = None):
        self.pallet = pallet

    def pack(self, boxes):
        best_solution = []
        best_score = 0
        best_unplaced = []

        all_permutations = list(itertools.permutations(boxes))

        # outer = tqdm(all_permutations, desc="全排列搜索", unit="perm", position=0)
        for perm in all_permutations:
            placed = []
            free_spaces = [(0, 0, 0)]

            for box in perm:
                placed_flag = False
                for orientation in box.orientations():
                    bl, bw, bh = orientation
                    for i, (x, y, z) in enumerate(free_spaces):
                        if (x + bl <= self.pallet.l and y + bw <= self.pallet.w and z + bh <= self.pallet.h and
                            not check_available(x, y, z, bl, bw, bh, placed)):
                            placed.append((x, y, z, bl, bw, bh, box.key()))
                            free_spaces.append((x + bl, y, z))
                            free_spaces.append((x, y + bw, z))
                            free_spaces.append((x, y, z + bh))
                            del free_spaces[i]
                            placed_flag = True
                            break
                    if placed_flag:
                        break
                if not placed_flag:
                    break  # 本排列失败，提前退出

            # 综合评估：利用率得分
            if placed:
                box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
                max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
                pallet_base = self.pallet.l * self.pallet.w
                total_volume = self.pallet.l * self.pallet.w * self.pallet.h
                used_volume = pallet_base * max_height
                util_overall = box_volume / total_volume
                util_used = box_volume / used_volume
                score = 0.5 * util_overall + 0.5 * util_used
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_solution = placed
                best_unplaced = [b for b in boxes if b.key() not in [p[6] for p in placed]]

        # outer.close()

        self.pallet.boxes = best_solution
        return best_solution, best_unplaced



class GeneticPacker:
    def __init__(self, pallet, packer,population_size=100, generations=100, mutation_rate=0.2):
        self.pallet = pallet
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.packer  = packer

    def pack(self, boxes):
        def fitness(order):
            pallet_tmp = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
            packer = self.packer(pallet_tmp)
            placed, _ = packer.pack(order)
            if not placed:
                return 0, placed

            box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
            max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
            total_volume = pallet_tmp.l * pallet_tmp.w * pallet_tmp.h
            used_volume = pallet_tmp.l * pallet_tmp.w * max_height

            util_overall = box_volume / total_volume
            util_used = box_volume / used_volume

            # 综合评分（可调节权重）
            score =  util_used
            return score, placed
        

        def crossover(p1, p2):
            cut = random.randint(1, len(p1) - 2)
            child = p1[:cut] + [b for b in p2 if b not in p1[:cut]]
            return child

        def mutate(order):
            a, b = random.sample(range(len(order)), 2)
            order[a], order[b] = order[b], order[a]

        # 初始化种群，每个个体是 Box 实例的排列
        population = [random.sample(boxes, len(boxes)) for _ in range(self.population_size)]
        best_solution = []
        best_score = 0
        best_placed = []

        for _ in range(self.generations):
            scored = []
            for ind in population:
                vol, placed = fitness(ind)
                scored.append((vol, placed, ind))
            scored.sort(key=lambda x: x[0], reverse=True)

            if scored[0][0] > best_score:
                best_score = scored[0][0]
                best_placed = scored[0][1]
                best_solution = scored[0][2]

            parents = [ind for _, _, ind in scored[:self.population_size // 2]]
            next_gen = []
            while len(next_gen) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    mutate(child)
                next_gen.append(child)
            population = next_gen

        # 最优个体重新打包
        final_pallet = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
        final_packer = self.packer(final_pallet)
        placed, unplaced = final_packer.pack(best_solution)
        self.pallet.boxes = placed
        return placed, unplaced

class SimulatedAnnealingPacker:
    def __init__(self, pallet, packer,initial_temp=100000, final_temp=1, alpha=0.95, max_iter=500):
        self.pallet = pallet
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha  # 温度衰减因子
        self.max_iter = max_iter
        self.packer = packer

    def pack(self, boxes):
        def fitness(order):
            pallet_tmp = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
            packer = self.packer(pallet_tmp)
            placed, _ = packer.pack(order)
            if not placed:
                return 0, placed

            box_volume = sum(l * w * h for _, _, _, l, w, h, _ in placed)
            max_height = max((z + h) for _, _, z, _, _, h, _ in placed)
            total_volume = pallet_tmp.l * pallet_tmp.w * pallet_tmp.h
            used_volume = pallet_tmp.l * pallet_tmp.w * max_height

            util_overall = box_volume / total_volume
            util_used = box_volume / used_volume

            # 综合评分（可调节权重）
            score =  util_used
            return score, placed

        def swap(box_list):
            new_list = box_list.copy()
            a, b = random.sample(range(len(new_list)), 2)
            new_list[a], new_list[b] = new_list[b], new_list[a]
            return new_list

        current_order = boxes[:]
        current_score, current_placed = fitness(current_order)
        best_order = current_order[:]
        best_score = current_score

        T = self.initial_temp
        iter_count = 0

        while T > self.final_temp and iter_count < self.max_iter:
            new_order = swap(current_order)
            new_score, _ = fitness(new_order)

            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_order = new_order
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_order = new_order

            T *= self.alpha
            iter_count += 1

        final_pallet = Pallet(self.pallet.l, self.pallet.w, self.pallet.h)
        final_packer = self.packer(final_pallet)
        placed, unplaced = final_packer.pack(best_order)
        self.pallet.boxes = placed
        return placed, unplaced
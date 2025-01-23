import json
from utils.dataset import Dataset
import numpy as np
import os
import copy


def read_file(num):
    with open(f"./outputs/Crossword_{num}", "r") as f:
        lines = [i.strip() for i in f.readlines()]
    idx = max(loc for loc, val in enumerate(lines) if val == "------ Current Answers --------")
    depth = int(lines[idx-4].split("=")[-1].strip())
    acc = float(lines[idx-3].split("=")[-1].strip())
    cacc = float(lines[idx-2].split("=")[-1].strip())
    answers = {}
    for i in lines[idx+1:-1]:
        key = i.split(":")[0].strip()
        val = i.split(":")[1].strip()
        answers[key] = val
    return answers, depth, acc, cacc

def read_files():
    ff = [file for file in os.listdir("./outputs") if "Crossword" in file]
    results = {}
    for f in ff:
        num = f.split("_")[-1]
        results[num] = read_file(num)[0]
    return results

def grid_to_array(board):
    res = np.zeros((15, 15), dtype = object)
    for row, line in enumerate(board):
        elems = [-1 if c.strip() in ["X",""] else int(c.strip()) for c in line.split("|")]
        res[row] = np.array(elems)
    return res

def fill_grid(grid, answers):
    filled_grid = copy.deepcopy(grid)
    for key, value in answers.items():
        num = int(key.split("_")[0])
        x, y = np.where(grid == num)
        x, y = x.item(), y.item()
        direction = key.split("_")[1]
        for i in range(len(value)):
            if direction == "d":
                filled_grid[x+i][y] = value[i]
            else:
                filled_grid[x][y+i] = value[i]
    return filled_grid


def char_analysis(dataset, threshold):
    results = read_files()
    counter = 0
    acc = []
    for data in dataset:
        grid = grid_to_array(data.board)
        gt_ans = {i:data.clues[i].answer for i in data.clues.keys()}
        gt_grid = fill_grid(grid, gt_ans)
        pred_grid = fill_grid(grid, results[str(data.number)])
        black = np.sum(np.where(gt_grid == -1, 1, 0))
        denominator = 15*15 - black
        numerator = np.sum(np.where(gt_grid == pred_grid, 1, 0)) - black
        char_acc = np.round(numerator/denominator, 2)
        acc.append(char_acc)
        if char_acc >= threshold:
            counter+=1
    return counter, np.mean(acc), np.std(acc)

def clue_analysis():
    ff = [file for file in os.listdir("./outputs") if "Crossword" in file]
    results = []
    for f in ff:
        num = f.split("_")[-1]
        results.append(read_file(num)[2])
    return np.mean(results), np.std(results)

def base_clue_analysis(dataset):
    res = []
    for data in dataset:
        with open(f"./outputs/Crossword_{data.number}", "r") as f:
            lines = [i.strip() for i in f.readlines()]
        idx = min(loc for loc, val in enumerate(lines) if val == "------ Current Answers --------")
        sidx = min(loc for loc, val in enumerate(lines[idx:]) if val == "-------------------------------")
        answers = {}
        for i in lines[idx+1:idx+sidx]:
            key = i.split(":")[0].strip()
            val = i.split(":")[1].strip()
            answers[key] = val
        counter = 0
        for key in data.clues.keys():
            gt_answer = data.clues[key].answer
            pred = answers.get(key, "")
            counter += 1 if gt_answer == pred else 0
        res.append(counter/len(data.clues))
    return np.mean(res), np.std(res)

def cacc():
    ff = [file for file in os.listdir("./outputs") if "Crossword" in file]
    results = []
    for f in ff:
        num = f.split("_")[-1]
        results.append(read_file(num)[-1])
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    dataset = Dataset()
    acc_base, std_base = base_clue_analysis(dataset)
    acc_sweep, std_sweep = clue_analysis()
    so50, mean, std = char_analysis(dataset, 0.5)
    cacc_mean, cacc_std = cacc(dataset)
    so90, _, _ = char_analysis(dataset, 0.9)
    so98, _, _ = char_analysis(dataset, 0.98)
    so1, _, _ = char_analysis(dataset, 1)

    print(
            f"Base clue accuracy {acc_base} ± {std_base}\n"\
            f"Final clue accuracy {acc_sweep} ± {std_sweep}\n"
            f"Final char accuracy {cacc_mean} ± {cacc_std}\n"
            f"Crosswords solved with 100% accuracy {so1}\n"
            f"Crosswords solved with >= 98% accuracy {so98}\n"
            f"Crosswords solved with >= 90% accuracy {so90}\n"
            f"Crosswords solved with >= 50% accuracy {so50}\n"
        )

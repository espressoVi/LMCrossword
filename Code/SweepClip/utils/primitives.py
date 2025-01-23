import json
import toml
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import copy
from utils.Prompt import Prompter, config
from itertools import chain


class Clue:
    def __init__(self,
                    position,
                    orientation,
                    clue_text,
                    answer,
                    constraints,
        ):
        self.position = position
        assert orientation in ["d", "a"]
        self.orientation = orientation
        self.clue_text = clue_text.strip()
        self.answer = answer
        self.constraints = constraints
    def __len__(self):
        return len(self.answer)
    def __repr__(self):
        return f"Clue: {self.clue_text} ({len(self)})"
    @property
    def num(self):
        return f"{self.position}_{self.orientation}"
    @property
    def flipped(self):
        return "a" if self.orientation == "d" else "d"
    @property
    def size(self):
        return len(self.constraints)


class Crossword:
    def __init__(self, number, clues, board):
        self.number = number
        self.board = board
        self.grid = self._grid_to_array(self.board)
        self.clues = self._parse(clues)
        self._create_graph()
        self.prompter = Prompter(config['model']['few_shot'])
        self.max_depth = 30
        self.current_answers = {}

    @staticmethod
    def _grid_to_array(board):
        res = np.zeros((config['dim'], )*2, dtype = object)
        for row, line in enumerate(board):
            elems = [-1 if c.strip() == "X" else 0 if c.strip() == "" else int(c.strip()) for c in line.split("|")]
            res[row] = np.array(elems)
        return res

    @staticmethod
    def _fill_grid(grid, answers):
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


    def _parse(self, clues):
        results = {}
        for clue in clues:
            #       {which clue     :(my_idx    ,   their_idx)}
            const = {i['depends_on']:(i['position'], i['idx'])\
                     for i in clue['constraints']}
            c = Clue(
                    position = clue['number'],
                    orientation = clue['orientation'][0],
                    clue_text = clue['clue'],
                    answer = clue['answer'],
                    constraints = const,
            )
            results[c.num] = c
        # Add forking constraints (i.e. 1 down and 1 across)
        for key, clue in results.items():
            nn = int(key.split(config["sep"])[0])
            if f"{nn}_{clue.flipped}" in results:
                clue.constraints[nn] = (0,0)
        return results

    def __repr__(self):
        return "\n".join(self.board)

    def _check_constraint(self, key1, answer1, key2, answer2):
        pos1, pos2 = self.clues[key1].constraints[int(key2.split(config['sep'])[0])]
        return answer1[pos1] == answer2[pos2]

    def _create_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.clues.keys())
        for clue in self.clues.values():
            for k in clue.constraints.keys():
                self.G.add_edge(clue.num, f"{k}_{clue.flipped}")

    def solve(self, LLM):
        self.model = LLM
        self.model.reset_cost()
        budget_exceeded = False
        self._get_init_candidates()
        for self.depth in range(self.max_depth):
            self._length_prune()
            search_next, solved = self._graph_prune()
            acc, cacc, surv, nex = self.write_output()
            print(
                    f"Current Depth: {self.depth}\n"\
                    f"Accuracy : {acc:.2f} | "\
                    f"Character Accuracy : {cacc:.2f} | "\
                    f"Current Filled: {surv}/{len(self.clues)}\n"\
                    f"Current filled: {np.sum(np.where(self._fill_grid(self.grid, self.current_answers) != 0, 1, 0))}\n"\
                    f"Next Search Over: {nex}\n"\
                    "----------------------------------"
            )
            if solved or budget_exceeded:
                break
            budget_exceeded = self.get_candidates(search_next)
    
    def _get_init_candidates(self):
        queries = []
        for cl in self.clues.keys():
            clue = self.clues[cl]
            clx = f"Clue: {clue.clue_text} ({len(clue)})"
            queries.append(clx)
        for key, clx in tqdm(zip(self.clues.keys(), queries), desc = "Initial candidates"):
            prompt = self.prompter.create_prompt(clx)
            response, Stop = self.model(prompt)
            self.current_answers[key] = response[0]

    def _length_prune(self):
        remove = set()
        for key, ans in self.current_answers.items():
            if len(ans) > len(self.clues[key]):
                self.current_answers[key] = ans[:len(self.clues[key])]
            elif len(ans) < len(self.clues[key]):
                remove.add(key)
        self.current_answers = {key:value for key, value in self.current_answers.items() if key not in remove}

    def _graph_prune(self):
        curr = dict(self.current_answers)
        pg, ng = nx.Graph(), nx.Graph()
        pg.add_nodes_from(list(self.current_answers.keys()))
        ng.add_nodes_from(list(self.current_answers.keys()))
        for key, answer in self.current_answers.items():
            cnf = set(v for v in self.G.neighbors(key) if v in self.current_answers)
            for cc in cnf:
                if self._check_constraint(key, answer, cc, curr[cc]):
                    pg.add_edge(key, cc)
                else:
                    ng.add_edge(key, cc)
        try:
            largest_cc = max(nx.connected_components(pg), key = len)
        except ValueError:
            largest_cc = set(np.random.choice(list(self.current_answers.keys()), 10, replace = False).tolist())
        conflict_subgraph = ng.subgraph(largest_cc).copy()
        while (
            (vert := max(conflict_subgraph.degree(conflict_subgraph.nodes), key = lambda x:x[1],))[1] > 0
        ):
            conflict_subgraph.remove_node(vert[0])
        survivors = set(conflict_subgraph.nodes)
        self.current_answers = {key:ans for key, ans \
                in self.current_answers.items() if key in survivors}
        ngbd = set()
        for key in survivors:
            ngbd = ngbd.union(set(self.G.neighbors(key)))
        ngbd = set(ngbd) - survivors
        self.next_search = len(ngbd)
        return ngbd, len(self.current_answers) == len(self.clues)

    def get_candidates(self, search_next):
        queries = []
        for cl in search_next:
            clue = self.clues[cl]
            hint = [config['sep']]*len(clue)
            for i in clue.constraints.keys():
                mkey = f"{i}_{clue.flipped}"
                if mkey in self.current_answers:
                    myidx, theiridx = clue.constraints[i]
                    hint[myidx] = self.current_answers[mkey][theiridx].upper()
            hint = " ".join(hint)
            clx = f"Clue: {clue.clue_text} ({len(clue)})"
            queries.append((clx, hint))
        stops = []
        for key, (clx, ht) in tqdm(zip(search_next, queries), desc = "LLM"):
            prompt = self.prompter.create_prompt(clx, ht)
            response, Stop = self.model(prompt)
            stops.append(Stop)
            self.current_answers[key] = response[0]
        return any(stops)

    def char_acc(self):
        gt_ans = {i:self.clues[i].answer for i in self.clues.keys()}
        gt_grid = self._fill_grid(self.grid, gt_ans)
        pred_grid = self._fill_grid(self.grid, self.current_answers)
        black = np.sum(np.where(gt_grid == -1, 1, 0))
        denominator = config['dim']**2 - black
        numerator = np.sum(np.where(gt_grid == pred_grid, 1, 0)) - black
        char_acc = np.round(numerator/denominator, 2)
        return char_acc

    def write_output(self):
        correct_gen = sum([1 for i, ans in self.clues.items() if i in self.current_answers and self.current_answers[i] == ans.answer])
        acc = correct_gen/(len(self.clues))
        char_acc = self.char_acc()
        surviving = len(self.current_answers)
        next_search = self.next_search
        with open(f"./outputs/Crossword_{str(self.number)}", "a") as f:
            f.write("-------------------------------\n")
            f.write(f"Current Depth = {self.depth}  \n")
            f.write(f"Acc = {acc:.2f}               \n")
            f.write(f"Char Acc = {char_acc:.2f}     \n")
            f.write("-------------------------------\n")
            f.write("------ Current Answers --------\n")
            f.write("\n".join([f"{k}: {v}" for k,v in self.current_answers.items()]))
            f.write("\n-------------------------------\n")
        return acc, char_acc, surviving, next_search

"""题四独立参考的代价方向、完整博弈与逐步收益校验。"""

from itertools import permutations
import json
from pathlib import Path
import random
import unittest

from experiments.treasure_reference import action_ledger, grid_weights, reference_distances, reference_game, validate_path
from src.problem4_treasure import evaluate_treasure


ROOT = Path(__file__).resolve().parent.parent


class TreasureReferenceTests(unittest.TestCase):
    def test_entering_cost_is_directed_and_goal_is_not_free(self):
        grid = [["A", 5, "B"]]
        positions = {"A": [0,0], "B": [0,2], "middle": [0,1]}
        matrix = reference_distances(grid, positions)
        self.assertEqual(matrix['A']['B'],6)
        self.assertEqual(matrix['A']['middle'],5)
        self.assertEqual(matrix['middle']['A'],1)
        self.assertEqual(validate_path(grid,[0,0],[0,2],[[0,0],[0,1],[0,2]],6),6)
        with self.assertRaises(ValueError):validate_path(grid,[0,0],[0,2],[[0,0],[0,1],[0,2]],5)
        self.assertEqual(reference_distances([['A',0,'B']],{'A':[0,0],'B':[0,2]})['A']['B'],1)
        with self.assertRaises(ValueError):reference_distances([['A','X','B']],{'A':[0,0],'B':[0,2]})

    def test_official_distances_depth_values_and_regret(self):
        case=json.loads((ROOT/'cases/problem4.json').read_text(encoding='utf8'))
        matrix=reference_distances(case['grid'],case['positions'])
        self.assertEqual(list(matrix['A'].values()),[0,12,10,4,8,8])
        expected=[[-1,2,-1,-1],[-4,-1,-2,-4],[-4,2,-1,-4],[-5,1,3,-7]]
        for h,values in enumerate(expected,1):
            ref=reference_game(matrix,case['treasure_values'],h)
            self.assertEqual(list(ref.action_values.values()),values)
            self.assertEqual(ref.best_targets,['T2'] if h<4 else ['T3'])
            self.assertEqual(ref.node_count,[5,17,41,65][h-1])
            self.assertEqual(ref.leaf_count,[4,12,24,24][h-1])
        full=reference_game(matrix,case['treasure_values'],4)
        self.assertEqual(full.principal_variation,['T3','T1','T2','T4'])
        self.assertEqual(full.value-full.action_values['T2'],2)
        ledger=action_ledger(matrix,case['treasure_values'],full.principal_variation)
        self.assertEqual([r['gain'] for r in ledger],[-1,1,0,-5])
        self.assertEqual((ledger[-1]['score_a'],ledger[-1]['score_b'],ledger[-1]['utility']),(-1,-4,3))

    def test_random_directed_games_against_separate_recursive_model(self):
        rng=random.Random(9341)
        for index in range(36):
            n=1+index%4;names=['A','B']+[f'T{i+1}' for i in range(n)]
            matrix={a:{b:0 if a==b else rng.randrange(10) for b in names} for a in names}
            values={t:rng.randrange(-5,10) for t in names[2:]}
            def solve(a,b,left,sa,sb,turn,h):
                if h==0 or not left:return sa-sb
                options=[]
                for target in left:
                    remain=tuple(t for t in left if t!=target)
                    if turn==0:options.append(solve(target,b,remain,sa+values[target]-matrix[a][target],sb,1,h-1))
                    else:options.append(solve(a,target,remain,sa,sb+values[target]-matrix[b][target],0,h-1))
                return (max if turn==0 else min)(options)
            for h in (0,1,n,n+1):
                ref=reference_game(matrix,values,h)
                self.assertEqual(ref.value,solve('A','B',tuple(values),0,0,0,h))
                for pruning in (False,True):
                    actual=evaluate_treasure(matrix,values,h,use_alpha_beta=pruning)
                    self.assertEqual(actual.best_value,ref.value)
                    self.assertIn(actual.best_target,ref.best_targets if h else [None])
                for target,sequence in ref.action_variations.items():
                    self.assertEqual(sequence[0],target)
                    self.assertEqual(len(sequence),min(h,n))
                    self.assertEqual(action_ledger(matrix,values,sequence)[-1]['utility'],ref.action_values[target])

    def test_negative_ties_zero_depth_and_invalid_values(self):
        labels=['A','B','T1','T2'];m={a:{b:0 if a==b else 1 for b in labels} for a in labels}
        ref=reference_game(m,{'T1':-2,'T2':-2},1)
        self.assertEqual((ref.value,ref.best_targets),(-3,['T1','T2']))
        self.assertEqual(reference_game(m,{},3).value,0)
        self.assertEqual(reference_game(m,{'T1':-2},0).principal_variation,[])
        for h in (-1,True,1.5):
            with self.assertRaises(ValueError):reference_game(m,{'T1':1},h)
        with self.assertRaises(ValueError):reference_game(m,{'T1':True},1)
        with self.assertRaises(ValueError):action_ledger(m,{'T1':1},['T1','T1'])
        for grid in ([],[[True]],[[-1]],[[1],[1,2]]):
            with self.assertRaises(ValueError):grid_weights(grid)


if __name__=='__main__':unittest.main()

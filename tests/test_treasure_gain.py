"""即时收益排序：双方局面、原求解器对照与实验数据完整性。"""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import unittest

from experiments.treasure_gain import evaluate_gain_treasure
from experiments.treasure_reference import reference_distances, reference_game
from src.problem4_treasure import evaluate_treasure
from tests.test_maze_experiment import workspace_temporary_directory


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'cases/problem4.json'
COUNTERS = ('node_visits', 'leaf_evals', 'cutoff_events', 'pruned_branches')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class DynamicGainTests(unittest.TestCase):
    def test_gain_order_on_both_players_negative_gains_and_stable_ties(self):
        names = ('A', 'B', 'T1', 'T2', 'T3')
        matrix = {a: {b: (0 if a == b else 1) for b in names} for a in names}
        matrix['A'].update(T1=5, T2=5, T3=8)
        matrix['B'].update(T1=1, T2=6, T3=2)
        values = {'T1': 0, 'T2': 0, 'T3': 0}
        result = evaluate_gain_treasure(matrix, values, 3, ordering='gain',
                                       use_alpha_beta=False, trace=True)
        orders = {tuple(event['path']): event for event in result.trace if event['event'] == 'order'}
        self.assertEqual(orders[()]['player'], 'A')
        self.assertEqual(orders[()]['candidates'], [
            {'target': 'T1', 'gain': -5}, {'target': 'T2', 'gain': -5},
            {'target': 'T3', 'gain': -8}])
        self.assertEqual(orders[('T1',)]['player'], 'B')
        self.assertEqual(orders[('T1',)]['candidates'], [
            {'target': 'T3', 'gain': -2}, {'target': 'T2', 'gain': -6}])
        self.assertEqual(result.best_value, reference_game(matrix, values, 3).value)

    def test_random_games_and_visited_order_against_independent_reference(self):
        rng = random.Random(79161)
        for sample in range(32):
            count = 1 + sample % 4
            names = ['A', 'B'] + [f'T{i + 1}' for i in range(count)]
            matrix = {a: {b: 0 if a == b else rng.randrange(8) for b in names} for a in names}
            values = {t: rng.randrange(-4, 7) for t in names[2:]}
            for depth in (0, 1, 2, count, count + 1):
                reference = reference_game(matrix, values, depth)
                for pruning in (False, True):
                    production = evaluate_treasure(matrix, values, depth, use_alpha_beta=pruning)
                    for ordering in ('original', 'gain'):
                        with self.subTest(sample=sample, depth=depth, pruning=pruning, order=ordering):
                            result = evaluate_gain_treasure(matrix, values, depth, ordering=ordering,
                                                            use_alpha_beta=pruning, trace=True)
                            self.assertEqual(result.best_value, reference.value)
                            self.assertIn(result.best_target, reference.best_targets if depth else [None])
                            if ordering == 'original':
                                self.assertEqual(result.best_target, production.best_target)
                                self.assertEqual(tuple(getattr(result.stats, f) for f in COUNTERS),
                                                 tuple(getattr(production.stats, f) for f in COUNTERS))
                            event_counts = Counter(event['event'] for event in result.trace)
                            self.assertEqual(result.stats.node_visits, event_counts['visit'])
                            self.assertEqual(result.stats.leaf_evals, event_counts['evaluate'])
                            self.assertEqual(result.stats.cutoff_events, event_counts['cutoff'])
                            self.assertEqual(result.stats.pruned_branches,
                                sum(len(e['skipped_targets']) for e in result.trace if e['event'] == 'cutoff'))
                            actions = defaultdict(list)
                            for event in result.trace:
                                if event['event'] == 'action':
                                    actions[tuple(event['path'])].append(event['target'])
                            for event in result.trace:
                                if event['event'] != 'order':
                                    continue
                                path = tuple(event['path'])
                                remaining = [t for t in values if t not in path]
                                source = event['source']
                                expected = remaining if ordering == 'original' else sorted(
                                    remaining, key=lambda t: values[t] - matrix[source][t], reverse=True)
                                self.assertEqual(event['candidates'],
                                    [{'target': t, 'gain': values[t] - matrix[source][t]} for t in expected])
                                self.assertEqual(actions[path], expected[:len(actions[path])])

    def test_official_counts_and_full_short_experiment(self):
        from experiments.exp4_gain import run_experiment
        from experiments.analyze_treasure_gain import analyze_results

        case = json.loads(INPUT.read_text(encoding='utf8'))
        matrix = reference_distances(case['grid'], case['positions'])
        for order, expected in (('original', (47, 16, 5, 6)), ('gain', (37, 12, 6, 8))):
            actual = evaluate_gain_treasure(matrix, case['treasure_values'], 4, ordering=order)
            self.assertEqual((actual.best_target, actual.best_value), ('T3', 3))
            self.assertEqual(tuple(getattr(actual.stats, f) for f in COUNTERS), expected)

        with workspace_temporary_directory() as tmp:
            output = tmp / 'result'
            for args in ({'batches': 0}, {'warmup': True}, {'max_loops': 0},
                         {'target_batch_ms': float('nan')}, {'max_batch_ms': 0}, {'analyze': 1}):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    run_experiment(output, **{'analyze': False, **args})
                self.assertFalse(output.exists())
            config = run_experiment(output, batches=2, warmup=1, max_loops=1,
                                    target_batch_ms=.01, analyze=False)
            self.assertEqual(config['status'], 'completed')
            self.assertEqual((output / 'input/problem4.json').read_bytes(), INPUT.read_bytes())
            self.assertEqual(len(read_csv(output / 'proof.csv')), 3)
            rows = read_csv(output / 'raw.csv')
            self.assertEqual(len(rows), 4)
            self.assertEqual([r['variant'] for r in rows], ['original', 'gain', 'gain', 'original'])
            self.assertTrue(all(r['best_target'] == 'T3' and r['best_value'] == '3' for r in rows))
            before = (output / 'raw.csv').read_bytes()
            with self.assertRaises(ValueError):
                run_experiment(output, batches=1, warmup=1, max_loops=1, analyze=False)
            self.assertEqual((output / 'raw.csv').read_bytes(), before)
            analysis = analyze_results(output, make_plots=False)
            self.assertEqual(len(read_csv(output / 'gain_summary.csv')), 2)
            self.assertAlmostEqual(analysis['comparison']['node_reduction'], 10 / 47)
            self.assertIn('total_ms_paired_speedup_median', analysis['comparison'])
            for change in ('node_visits', 'best_value', 'missing', 'duplicate_position', 'nan_time'):
                bad = [dict(row) for row in rows]
                if change == 'node_visits':
                    bad[0]['node_visits'] = str(int(bad[0]['node_visits']) + 1)
                elif change == 'best_value':
                    bad[0]['best_value'] = '999'
                elif change == 'missing':
                    bad.pop()
                elif change == 'duplicate_position':
                    bad[1]['order_in_pair'] = bad[0]['order_in_pair']
                else:
                    bad[0]['total_ms_mean'] = 'nan'
                with (output / 'raw.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(bad)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    analyze_results(output, make_plots=False)
                (output / 'raw.csv').write_bytes(before)


if __name__ == '__main__':
    unittest.main()

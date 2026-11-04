"""题四固定全局优先级：逐层访问顺序、独立结果与实验数据完整性。"""

from collections import Counter, defaultdict
import csv
from itertools import permutations
import json
from pathlib import Path
import unittest

from experiments.treasure_reference import reference_distances, reference_game
from src.problem4_treasure import evaluate_treasure
from tests.test_maze_experiment import workspace_temporary_directory


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'cases' / 'problem4.json'
# Independent alpha-beta audit of all fixed priorities on the official case.
EXPECTED_NODES = dict(zip(
    ('1234', '1243', '1324', '1342', '1423', '1432',
     '2134', '2143', '2314', '2341', '2413', '2431',
     '3124', '3142', '3214', '3241', '3412', '3421',
     '4123', '4132', '4213', '4231', '4312', '4321'),
    (47, 47, 42, 42, 59, 52, 39, 39, 37, 37, 41, 39,
     31, 31, 31, 31, 35, 35, 65, 61, 60, 51, 46, 46),
))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class FixedPriorityTests(unittest.TestCase):
    def test_invalid_parameters_do_not_create_results(self):
        from experiments.exp4_order import run_experiment

        with workspace_temporary_directory() as tmp:
            output = tmp / 'result'
            for args in ({'batches': 0}, {'warmup': True}, {'max_loops': 0},
                         {'target_batch_ms': float('nan')}, {'max_batch_ms': 0}, {'analyze': 1}):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    run_experiment(output, **{'analyze': False, **args})
                self.assertFalse(output.exists())

            broken = json.loads(INPUT.read_text(encoding='utf8'))
            broken['positions']['T1'] = broken['positions']['A']
            source = tmp / 'bad.json'
            source.write_text(json.dumps(broken), encoding='utf8')
            with self.assertRaises(ValueError):
                run_experiment(output, input_path=source, analyze=False)
            self.assertFalse(output.exists())

    def test_all_orders_use_global_priority_at_every_visited_node(self):
        case = json.loads(INPUT.read_text(encoding='utf8'))
        values = case['treasure_values']
        matrix = reference_distances(case['grid'], case['positions'])
        reference = reference_game(matrix, values, 4)
        self.assertEqual((reference.value, reference.best_targets), (3, ['T3']))

        visited = Counter()
        for priority in permutations(values):
            ordered = {target: values[target] for target in priority}
            for pruning in (False, True):
                with self.subTest(priority=priority, pruning=pruning):
                    result = evaluate_treasure(matrix, ordered, 4,
                                               use_alpha_beta=pruning, trace=True)
                    self.assertEqual((result.best_value, result.best_target),
                                     (reference.value, reference.best_targets[0]))
                    counts = Counter(event['event'] for event in result.trace)
                    self.assertEqual(result.stats.node_visits, counts['visit'])
                    self.assertEqual(result.stats.leaf_evals, counts['evaluate'])
                    self.assertEqual(result.stats.cutoff_events, counts['cutoff'])
                    self.assertEqual(result.stats.pruned_branches,
                                     sum(len(e['skipped_targets']) for e in result.trace
                                         if e['event'] == 'cutoff'))
                    actions_by_path = defaultdict(list)
                    for event in result.trace:
                        if event['event'] == 'action':
                            actions_by_path[tuple(event['path'])].append(event['target'])
                    for path, actual in actions_by_path.items():
                        remaining = [t for t in priority if t not in path]
                        # Pruning may stop early, but cannot reorder the explored prefix.
                        self.assertEqual(actual, remaining[:len(actual)])
                        if not pruning:
                            self.assertEqual(actual, remaining)
                    if pruning:
                        key = ''.join(t[1:] for t in priority)
                        self.assertEqual(result.stats.node_visits, EXPECTED_NODES[key])
                        visited[result.stats.node_visits] += 1
                    else:
                        self.assertEqual((result.stats.node_visits, result.stats.leaf_evals),
                                         (65, 24))

        self.assertEqual(visited, {31: 4, 35: 2, 37: 2, 39: 3, 41: 1, 42: 2,
                                   46: 2, 47: 2, 51: 1, 52: 1, 59: 1, 60: 1,
                                   61: 1, 65: 1})

    def test_short_complete_run_and_output_protection(self):
        from experiments.exp4_order import run_experiment
        from experiments.analyze_treasure_order import analyze_results

        with workspace_temporary_directory() as tmp:
            output = tmp / 'result'
            config = run_experiment(output, batches=2, warmup=1, max_loops=1,
                                    target_batch_ms=.01, analyze=False)
            self.assertEqual(config['status'], 'completed')
            self.assertEqual((output / 'input' / 'problem4.json').read_bytes(), INPUT.read_bytes())
            rows = read_csv(output / 'raw.csv')
            self.assertEqual(len(rows), 50)
            self.assertEqual(len(read_csv(output / 'proof.csv')), 48)
            grouped = defaultdict(list)
            for row in rows:
                grouped[int(row['batch_index'])].append(row)
                self.assertEqual((row['best_target'], row['best_value'], row['reference_value'],
                                  row['correct']), ('T3', '3', '3', 'True'))
                self.assertEqual(row['batch_loops'], '1')
            for batch in grouped.values():
                self.assertEqual(len(batch), 25)
                self.assertEqual(len({(r['priority_id'], r['algorithm']) for r in batch}), 25)
                self.assertEqual(len({r['execution_position'] for r in batch}), 25)
                self.assertEqual(Counter(r['algorithm'] for r in batch),
                                 {'alpha_beta': 24, 'minimax': 1})
            before = (output / 'raw.csv').read_bytes()
            with self.assertRaises(ValueError):
                run_experiment(output, batches=1, warmup=1, max_loops=1, analyze=False)
            self.assertEqual((output / 'raw.csv').read_bytes(), before)

            analyze_results(output, make_plots=False)
            self.assertEqual(len(read_csv(output / 'order_summary.csv')), 25)
            self.assertEqual(json.loads((output / 'config.json').read_text(encoding='utf8'))
                             ['analysis_status'], 'completed')
            for change in ('node_visits', 'best_value', 'missing', 'duplicate_position', 'nan_time'):
                bad = [dict(row) for row in rows]
                if change == 'node_visits':
                    bad[0]['node_visits'] = str(int(bad[0]['node_visits']) + 1)
                elif change == 'best_value':
                    bad[0]['best_value'] = '999'
                elif change == 'missing':
                    bad.pop()
                elif change == 'duplicate_position':
                    same_batch = next(row for row in bad[1:]
                                      if row['batch_index'] == bad[0]['batch_index'])
                    same_batch['execution_position'] = bad[0]['execution_position']
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

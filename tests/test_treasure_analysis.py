"""题四分析对输入、计时范围、配对和决策质量的拒错检查。"""

from copy import deepcopy
import csv
import json
import unittest

from experiments.analyze_treasure_depths import _read_rows, _validate_batches, _validate_evidence, _verify_sources, summarize
from experiments.exp4_depths import run_experiment
from experiments.treasure_reference import reference_game
from tests.test_maze_experiment import workspace_temporary_directory
from tests.test_treasure_experiment import tiny_input


class TreasureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.context=workspace_temporary_directory();self.tmp=self.context.__enter__()
        self.source=self.tmp/'input.json';self.case=tiny_input(self.source);self.out=self.tmp/'results'
        self.config=run_experiment(self.out,input_path=self.source,depths=(1,2),batches=2,warmup=1,
                                   max_loops=1,target_batch_ms=.01,analyze=False)
        self.rows=_read_rows(self.out/'raw.csv')
    def tearDown(self):self.context.__exit__(None,None,None)

    def test_complete_pairs_and_summary_ratios(self):
        _validate_batches(self.rows,self.config)
        summary,paired=summarize(self.rows)
        self.assertEqual(len(summary),8);self.assertEqual(len(paired),4)
        lookup={(r['scope'],r['depth'],r['algorithm']):r for r in summary}
        for row in paired:
            mm,ab=(lookup[row['scope'],row['depth'],a] for a in ('minimax','alpha_beta'))
            self.assertAlmostEqual(row['node_reduction'],1-ab['node_visits']/mm['node_visits'])
            self.assertAlmostEqual(row['total_speedup'],mm['total_ms_median']/ab['total_ms_median'])
        missing=self.rows[:-1]
        with self.assertRaises(ValueError):_validate_batches(missing,self.config)
        mismatch=deepcopy(self.rows);mismatch[0]['batch_loops']=2
        with self.assertRaises(ValueError):_validate_batches(mismatch,self.config)

    def test_raw_rejects_scope_costs_nonfinite_and_duplicates(self):
        with (self.out/'raw.csv').open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
        for change in ({'correct':'False'},{'search_ms_mean':'nan'},{'total_ms_mean':'0'},
                       {'batch_loops':'0'},{'best_value':'999'},{'scope':'bad'}):
            bad=deepcopy(raw);bad[0].update(change);path=self.tmp/'bad.csv'
            with path.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=raw[0]);writer.writeheader();writer.writerows(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):_read_rows(path)
        duplicate=self.tmp/'duplicate.csv'
        with duplicate.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=raw[0]);writer.writeheader();writer.writerows(raw+[raw[0]])
        with self.assertRaises(ValueError):_read_rows(duplicate)

    def test_saved_evidence_and_source_hashes_are_checked(self):
        matrix=json.loads((self.out/'distance_matrix.json').read_text(encoding='utf8'))
        refs={d:reference_game(matrix,self.case['treasure_values'],d) for d in (1,2)}
        decisions=_validate_evidence(self.out,self.case,matrix,refs,self.rows,self.config)
        self.assertEqual(len(decisions),2)
        _verify_sources(self.out,self.config)
        bad=deepcopy(self.config);bad['source_baseline_sha256']='0'*64
        with self.assertRaises(ValueError):_verify_sources(self.out,bad)
        data=(self.out/'distance_matrix.json').read_bytes();matrix['A']['T1']+=1
        with self.assertRaises(ValueError):_validate_evidence(self.out,self.case,matrix,refs,self.rows,self.config)
        (self.out/'distance_matrix.json').write_bytes(data)
        path=self.out/'decisions.csv'
        with path.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
        records[0]['first_move_regret']='999'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
        matrix=json.loads(data)
        with self.assertRaises(ValueError):_validate_evidence(self.out,self.case,matrix,refs,self.rows,self.config)


if __name__=='__main__':unittest.main()

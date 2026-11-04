"""题四双范围计时、固定输入、副本和诊断证据的集成检查。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest
from zipfile import ZipFile

from experiments.exp4_depths import run_experiment
from tests.test_maze_experiment import workspace_temporary_directory


def tiny_input(path):
    case={'grid':[['A','T1'],['T2','B']],
          'positions':{'A':[0,0],'B':[1,1],'T1':[0,1],'T2':[1,0]},
          'treasure_values':{'T1':2,'T2':3}}
    path.write_text(json.dumps(case),encoding='utf8')
    return case


class TreasureExperimentTests(unittest.TestCase):
    def test_small_complete_run_has_both_scopes_and_exact_evidence(self):
        with workspace_temporary_directory() as tmp:
            source=tmp/'input.json';tiny_input(source);before=source.read_bytes();out=tmp/'results'
            c=run_experiment(out,input_path=source,depths=(1,2),batches=2,warmup=1,max_loops=1,
                             target_batch_ms=.01,analyze=False)
            self.assertEqual(c['status'],'completed');self.assertEqual(c['raw_rows'],16)
            self.assertEqual(c['total_timed_calls'],16)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual((out/'input/problem4.json').read_bytes(),before)
            with (out/'raw.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            pairs={}
            for r in rows:
                pairs.setdefault((r['scope'],r['depth'],r['batch_index']),[]).append(r)
                self.assertEqual(r['best_value'],r['reference_value'])
                self.assertEqual(r['correct'],'True')
                self.assertGreaterEqual(float(r['total_ms_mean']),float(r['distance_ms_mean'])+float(r['query_ms_mean']))
                self.assertEqual(float(r['distance_ms_mean'])==0,r['scope']=='precomputed')
            for pair in pairs.values():
                self.assertEqual({r['algorithm'] for r in pair},{'minimax','alpha_beta'})
                self.assertEqual({r['order_in_pair'] for r in pair},{'0','1'})
                self.assertEqual(len({r['batch_loops'] for r in pair}),1)
            with (out/'action_values.csv').open(encoding='utf-8-sig',newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))),4)
            with (out/'distance_audit.csv').open(encoding='utf-8-sig',newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))),16)
            self.assertEqual(len(json.loads((out/'paths.json').read_text(encoding='utf8'))),12)
            with ZipFile(out/'source_baseline.zip') as archive:
                self.assertEqual(set(archive.namelist()),set(c['source_sha256']))
                for name,h in c['source_sha256'].items():self.assertEqual(sha256(archive.read(name)).hexdigest(),h)
            self.assertEqual(sha256((out/'source_baseline.zip').read_bytes()).hexdigest(),c['source_baseline_sha256'])
            with self.assertRaises(ValueError):run_experiment(out,input_path=source,analyze=False)

    def test_invalid_parameters_or_input_do_not_create_output(self):
        with workspace_temporary_directory() as tmp:
            source=tmp/'input.json';tiny_input(source);out=tmp/'bad'
            for args in ({'depths':()},{'depths':(True,)},{'depths':(1,1)},{'batches':0},{'warmup':True},
                         {'max_loops':0},{'target_batch_ms':float('nan')},{'max_batch_ms':0},{'analyze':1}):
                with self.subTest(args=args),self.assertRaises(ValueError):
                    run_experiment(out,input_path=source,**{'analyze':False,**args})
                self.assertFalse(out.exists())
            bad=json.loads(source.read_text(encoding='utf8'));bad['positions']['T1']=[0,0]
            source.write_text(json.dumps(bad),encoding='utf8')
            with self.assertRaises(ValueError):run_experiment(out,input_path=source,analyze=False)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()

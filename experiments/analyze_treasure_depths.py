"""题四已保存数据的核验、汇总和图表，不重新采集时间。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from zipfile import ZipFile

from experiments.analyze_weighted_heuristics import distribution, _save, _setup_plotting
from experiments.exp3_minimax_alpha_beta import _write_csv, _write_json
from experiments.exp4_depths import ALGORITHMS, COUNTERS, SCOPES, TIMINGS, load_input
from experiments.treasure_reference import action_ledger, grid_weights, reference_distances, reference_game, validate_path


ROOT = Path(__file__).resolve().parent.parent
COLORS = {"minimax": "#315d7a", "alpha_beta": "#21847b"}
LABELS = {"minimax": "Minimax", "alpha_beta": "Alpha-Beta"}


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_rows(path):
    integers = ("depth", "batch_index", "batch_loops", "order_in_pair", "scope_order_in_batch",
                "best_value", "reference_value", *COUNTERS)
    times = (*(f"{t}_mean" for t in TIMINGS), "batch_total_ms")
    required = {"run_id", "input_file", "input_sha256", "scope", "algorithm", "best_target", "correct", *integers, *times}
    rows, seen = [], set()
    for raw in _csv(path):
        if not required <= set(raw):
            raise ValueError("原始记录字段不完整")
        row = dict(raw)
        for f in integers:
            row[f] = int(row[f])
        for f in times:
            row[f] = float(row[f])
            if not math.isfinite(row[f]) or row[f] < 0:
                raise ValueError("时间必须有限且非负")
        if row["scope"] not in SCOPES or row["algorithm"] not in ALGORITHMS:
            raise ValueError("未知计时范围或算法")
        if str(row["correct"]).lower() not in {"true", "1"} or row["best_value"] != row["reference_value"]:
            raise ValueError("根值未通过参考校验")
        row["correct"] = True
        if (row["depth"] < 1 or row["batch_index"] < 0 or row["batch_loops"] < 1
                or row["order_in_pair"] not in (0, 1) or row["scope_order_in_batch"] not in (0, 1)
                or any(row[f] < 0 for f in COUNTERS)
                or not 1 <= row["leaf_evals"] <= row["node_visits"]
                or row["cutoff_events"] > row["pruned_branches"]):
            raise ValueError("批次、执行顺序或访问计数无效")
        if any(row[f"{t}_mean"] <= 0 for t in ("search_ms", "query_ms", "total_ms")):
            raise ValueError("搜索、查询和总时间必须为正")
        if row["scope"] == "precomputed" and row["distance_ms_mean"] != 0:
            raise ValueError("矩阵复用模式不能重新收费寻路成本")
        if row["scope"] == "end_to_end" and row["distance_ms_mean"] <= 0:
            raise ValueError("完整模式必须包含新建距离矩阵")
        if (row["search_ms_mean"] > row["query_ms_mean"] + 1e-8
                or row["distance_ms_mean"] + row["query_ms_mean"] > row["total_ms_mean"] + 1e-8
                or not math.isclose(row["batch_total_ms"], row["total_ms_mean"] * row["batch_loops"], rel_tol=1e-8)):
            raise ValueError("时间范围或批时与均值矛盾")
        key = row["scope"], row["depth"], row["algorithm"], row["batch_index"]
        if key in seen:
            raise ValueError("重复批次")
        seen.add(key)
        rows.append(row)
    if not rows or len({r["run_id"] for r in rows}) != 1:
        raise ValueError("数据为空或混合多次运行")
    return rows


def _verify_sources(folder, config):
    hashes = config.get("source_sha256", {})
    required = {"src/problem4_treasure.py", "src/grid.py", "experiments/treasure_reference.py",
                "experiments/exp4_depths.py", "experiments/analyze_treasure_depths.py"}
    if not required <= set(hashes):
        raise ValueError("缺少本轮源码哈希")
    for name, expected in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"源码变化：{name}，请使用本轮源码归档")
    archive = folder / "source_baseline.zip"
    if sha256(archive.read_bytes()).hexdigest() != config.get("source_baseline_sha256"):
        raise ValueError("源码归档哈希不符")
    with ZipFile(archive) as stream:
        if set(stream.namelist()) != set(hashes) or len(stream.namelist()) != len(hashes):
            raise ValueError("源码归档成员不符")
        for name, expected in hashes.items():
            if sha256(stream.read(name)).hexdigest() != expected:
                raise ValueError("归档源码内容不符")


def _validate_batches(rows, config):
    expected = {(s, d, a, b) for s in SCOPES for d in config["depths"]
                for a in ALGORITHMS for b in range(config["batches"])}
    if {(r["scope"], r["depth"], r["algorithm"], r["batch_index"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("各深度双算法双范围的批次不完整")
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["depth"], row["scope"], row["batch_index"]].append(row)
        if any(row[f] != config[f] for f in ("run_id", "input_file", "input_sha256")):
            raise ValueError("原始输入或运行标识不符")
    for (depth, scope, batch), entries in pairs.items():
        if len({r["batch_loops"] for r in entries}) != 1 or {r["order_in_pair"] for r in entries} != {0, 1}:
            raise ValueError("同对两算法必须同调用次数并占据不同位置")
        index = config["depths"].index(depth)
        actual = [r["algorithm"] for r in sorted(entries, key=lambda r: r["order_in_pair"])]
        wanted = list(ALGORITHMS[::1 if (index * config["batches"] + batch) % 2 == 0 else -1])
        if actual != wanted or len({r["scope_order_in_batch"] for r in entries}) != 1:
            raise ValueError("执行顺序与配置不符")
        scopes = SCOPES[::1 if (index + batch) % 2 == 0 else -1]
        if entries[0]["scope_order_in_batch"] != scopes.index(scope):
            raise ValueError("计时范围先后顺序不符")
    for scope in SCOPES:
        for depth in config['depths']:
            if len({r['batch_loops'] for r in rows if r['scope']==scope and r['depth']==depth}) != 1:
                raise ValueError("同一深度和计时范围的循环数不应随批次变化")


def _validate_evidence(folder, case, matrix, refs, rows, config):
    independent = reference_distances(case["grid"], case["positions"])
    if matrix != independent or json.loads((folder / "dijkstra_reference.json").read_text(encoding="utf8")) != independent:
        raise ValueError("保存的距离矩阵与独立参考不符")
    audit = _csv(folder / "distance_audit.csv")
    wanted = {(s, t) for s in matrix for t in matrix}
    if len(audit) != len(wanted) or {(r['source'], r['target']) for r in audit} != wanted:
        raise ValueError("距离审计覆盖不完整")
    for r in audit:
        if str(r["correct"]).lower() != "true" or any(int(r[f]) != matrix[r['source']][r['target']] for f in ("astar_cost", "dijkstra_cost", "path_cost")):
            raise ValueError("距离审计数值不符")
    paths = json.loads((folder / "paths.json").read_text(encoding="utf8"))
    if len(paths) != len(matrix) * (len(matrix) - 1) or {(r['source'], r['target']) for r in paths} != {(s, t) for s, t in wanted if s != t}:
        raise ValueError("路径记录覆盖不完整")
    for item in paths:
        validate_path(case['grid'], case['positions'][item['source']], case['positions'][item['target']],
                      item['path'], matrix[item['source']][item['target']])
    weights=grid_weights(case['grid']);expected_steps={}
    for item in paths:
        cumulative=0
        for index,(row,col) in enumerate(item['path']):
            entering=0 if index==0 else weights[row][col];cumulative+=entering
            expected_steps[item['source'],item['target'],index]={'row':row,'col':col,'cell':case['grid'][row][col],
                                                               'entering_cost':entering,'cumulative_cost':cumulative}
    saved_steps=_csv(folder/'path_steps.csv');seen_steps=set()
    for r in saved_steps:
        key=r['source'],r['target'],int(r['step'])
        if key not in expected_steps or key in seen_steps or any(str(value)!=r[f] for f,value in expected_steps[key].items()):
            raise ValueError("路径逐格进入代价记录不符")
        seen_steps.add(key)
    if len(saved_steps)!=len(expected_steps):raise ValueError("路径逐格记录不完整")
    proof = _csv(folder / "proof.csv")
    lookup = {(int(r['depth']), r['algorithm']): r for r in proof}
    if len(lookup) != len(proof) or set(lookup) != {(d, a) for d in refs for a in ALGORITHMS}:
        raise ValueError("博弈证明覆盖不完整")
    for r in rows:
        ref = refs[r['depth']]
        record = lookup[r['depth'], r['algorithm']]
        if (str(record['correct']).lower()!='true' or r['best_target'] not in ref.best_targets or r['best_value'] != ref.value
                or r['reference_value'] != ref.value or record['best_target'] != r['best_target']
                or any(int(record[f]) != r[f] for f in ('best_value', 'reference_value', *COUNTERS))):
            raise ValueError("根决策或确定性计数与证明不符")
        if r['node_visits'] > ref.node_count or r['leaf_evals'] > ref.leaf_count:
            raise ValueError("访问多于完整博弈树")
        if r['algorithm'] == 'minimax' and (r['node_visits'] != ref.node_count or r['leaf_evals'] != ref.leaf_count
                or r['cutoff_events'] or r['pruned_branches']):
            raise ValueError("Minimax 的完整计数错误")
    full = reference_game(independent, case['treasure_values'], len(case['treasure_values']))
    actions = _csv(folder / 'action_values.csv')
    if len(actions) != len(refs) * len(case['treasure_values']) or {(int(r['depth']), r['target']) for r in actions} != {(d,t) for d in refs for t in case['treasure_values']}:
        raise ValueError("首步精确值覆盖不完整")
    for r in actions:
        if (int(r['exact_value']) != refs[int(r['depth'])].action_values[r['target']] or r['exact_value'] != r['minimax_child_value']
                or int(r['full_horizon_value'])!=full.action_values[r['target']]
                or (r['optimal_at_depth'].lower()=='true') != (r['target'] in refs[int(r['depth'])].best_targets)):
            raise ValueError("首步值不是独立精确值")
    decisions = _csv(folder / 'decisions.csv')
    if len(decisions) != len(refs) or {int(r['depth']) for r in decisions} != set(refs):
        raise ValueError("决策表深度不完整")
    for r in decisions:
        ref = refs[int(r['depth'])]
        if (r['chosen_target'] != ref.best_targets[0] or int(r['depth_value']) != ref.value
                or int(r['first_move_full_value']) != full.action_values[r['chosen_target']]
                or int(r['full_optimal_value']) != full.value
                or int(r['first_move_regret']) != full.value - full.action_values[r['chosen_target']]):
            raise ValueError("完整首步质量或 regret 错误")
    ledger = _csv(folder / 'action_ledger.csv')
    expected_ledgers = {}
    for d, ref in refs.items():
        for record in action_ledger(matrix, case['treasure_values'], ref.principal_variation):
            expected_ledgers['depth_optimal', d, record['step']] = record
    for target, sequence in full.action_variations.items():
        for record in action_ledger(matrix, case['treasure_values'], sequence):
            expected_ledgers[f'full_first_{target}', len(case['treasure_values']), record['step']] = record
    if len(ledger) != len(expected_ledgers):
        raise ValueError("收益路径表行数错误")
    seen = set()
    for r in ledger:
        key = r['scenario'], int(r['depth']), int(r['step'])
        if key in seen or key not in expected_ledgers:
            raise ValueError("收益路径重复或缺失")
        seen.add(key)
        if any(str(value) != r[field] for field, value in expected_ledgers[key].items()):
            raise ValueError("收益逐步累计错误")
    traces = json.loads((folder / 'search_traces.json').read_text(encoding='utf8'))
    variations=json.loads((folder/'principal_variations.json').read_text(encoding='utf8'))
    if variations!={str(d):asdict(ref) for d,ref in refs.items()}:
        raise ValueError("保存的主变化路径或首步真值与独立参考不符")
    if set(traces) != {f'H{d}_{a}' for d in refs for a in ALGORITHMS}:
        raise ValueError("轨迹覆盖不完整")
    for d in refs:
        for a in ALGORITHMS:
            item = traces[f'H{d}_{a}']; record = lookup[d,a]
            if item['best_value'] != refs[d].value or item['best_target'] not in refs[d].best_targets:
                raise ValueError("轨迹根值不符")
            events = item['trace']
            visits = [e for e in events if e['event']=='visit']
            leaves = [e for e in events if e['event']=='evaluate']
            cutoffs = [e for e in events if e['event']=='cutoff']
            counts = (len(visits),len(leaves),len(cutoffs),sum(len(e['skipped_targets']) for e in cutoffs))
            if counts != tuple(int(record[f]) for f in COUNTERS) or any(item['stats'][f] != int(record[f]) for f in COUNTERS):
                raise ValueError("轨迹计数不符")
            visited={tuple(e['path']) for e in visits}
            if len(visited)!=len(visits):raise ValueError("同一行动前缀在轨迹中重复访问")
            for event in visits:
                path=event['path'];steps=action_ledger(matrix,case['treasure_values'],path)
                state=event['state'];score_a=steps[-1]['score_a'] if steps else 0;score_b=steps[-1]['score_b'] if steps else 0
                pos_a=next((r['target'] for r in reversed(steps) if r['player']=='A'),'A')
                pos_b=next((r['target'] for r in reversed(steps) if r['player']=='B'),'B')
                if (len(path)>d or state['score_a']!=score_a or state['score_b']!=score_b
                        or (state['pos_a'],state['pos_b'])!=(pos_a,pos_b)
                        or state['turn']!=('A' if len(path)%2==0 else 'B')
                        or state['remaining']!=[t for t in case['treasure_values'] if t not in path]
                        or event['remaining_depth']!=d-len(path)):
                    raise ValueError("轨迹状态转移与独立行动回放不符")
            for event in leaves:
                steps=action_ledger(matrix,case['treasure_values'],event['path'])
                if tuple(event['path']) not in visited or event['value']!=(steps[-1]['utility'] if steps else 0):
                    raise ValueError("轨迹叶子收益不符")
            for event in cutoffs:
                prefix=tuple(event['path'])
                if event['alpha'] is None or event['beta'] is None or event['alpha']<event['beta']:
                    raise ValueError("轨迹截断条件不成立")
                if any(path[:len(prefix)+1]==prefix+(target,) for path in visited for target in event['skipped_targets']):
                    raise ValueError("声明剪掉的子树仍被访问")
    return decisions


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['scope'],row['depth'],row['algorithm']].append(row)
    result=[]
    for (scope,depth,algorithm), group in sorted(grouped.items()):
        first=group[0]
        if any(r[f]!=first[f] for r in group for f in ('best_target','best_value',*COUNTERS)):
            raise ValueError("批次间决策或计数不稳定")
        row={'scope':scope,'depth':depth,'algorithm':algorithm,'n_batches':len(group),'best_target':first['best_target'],
             'best_value':first['best_value'],**{f:first[f] for f in COUNTERS},
             'batch_loops_min':min(r['batch_loops'] for r in group),'batch_loops_max':max(r['batch_loops'] for r in group)}
        for t in TIMINGS:
            for key,value in distribution([r[f'{t}_mean'] for r in group]).items():row[f'{t}_{key}']=value
        result.append(row)
    lookup={(r['scope'],r['depth'],r['algorithm']):r for r in result}
    paired=[]
    for scope,depth in sorted({(r['scope'],r['depth']) for r in result}):
        mm,ab=(lookup[scope,depth,a] for a in ALGORITHMS)
        if mm['best_value']!=ab['best_value'] or ab['node_visits']>mm['node_visits'] or ab['leaf_evals']>mm['leaf_evals']:
            raise ValueError("双算法不等值或剪枝计数反常")
        pair={'scope':scope,'depth':depth,'node_reduction':1-ab['node_visits']/mm['node_visits'],
              'leaf_reduction':1-ab['leaf_evals']/mm['leaf_evals']}
        for name in ('search','query','total'):
            pair[f'{name}_speedup']=mm[f'{name}_ms_median']/ab[f'{name}_ms_median']
            pair[f'{name}_time_reduction']=1-ab[f'{name}_ms_median']/mm[f'{name}_ms_median']
        paired.append(pair)
    return result,paired


def make_figures(folder,case,matrix,refs,decisions,summary):
    plt=_setup_plotting();directory=folder/'figures';directory.mkdir(exist_ok=True)
    import numpy as np
    from matplotlib.patches import Rectangle
    # 地图和距离矩阵，地图轴 row 向下。
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.8));weights=grid_weights(case['grid'])
    display=np.array([[float('nan') if w is None else w for w in row] for row in weights])
    cmap=plt.get_cmap('YlOrBr').copy();cmap.set_bad('#384451')
    axes[0].imshow(display,cmap=cmap,vmin=0,vmax=max(w for row in weights for w in row if w is not None))
    for r,row in enumerate(case['grid']):
        for c,cell in enumerate(row):
            if cell!='X':axes[0].text(c,r,str(cell),ha='center',va='center',fontsize=10)
    axes[0].set_xticks(range(len(weights[0])));axes[0].set_yticks(range(len(weights)))
    axes[0].set_title('Official weighted map: marked cells cost 1')
    names=list(matrix);data=np.array([[matrix[a][b] for b in names] for a in names])
    axes[1].imshow(data,cmap='Blues');axes[1].set_xticks(range(len(names)),names);axes[1].set_yticks(range(len(names)),names)
    for r in range(len(names)):
        for c in range(len(names)):axes[1].text(c,r,str(data[r,c]),ha='center',va='center',color='white' if data[r,c]>data.max()*.65 else '#243747')
    axes[1].set_title('A* distances, independently checked by Dijkstra');axes[1].set_xlabel('Target');axes[1].set_ylabel('Source')
    fig.tight_layout();_save(fig,directory,'01_map_distances');plt.close(fig)
    depths=sorted(refs);targets=list(case['treasure_values'])
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.8));values=np.array([[refs[d].action_values[t] for t in targets] for d in depths])
    axes[0].imshow(values,cmap='RdYlGn',aspect='auto');axes[0].set_xticks(range(len(targets)),targets);axes[0].set_yticks(range(len(depths)),[f'H={d}' for d in depths])
    for i,d in enumerate(depths):
        for j,t in enumerate(targets):
            axes[0].text(j,i,str(values[i,j]),ha='center',va='center',fontsize=12)
            if t in refs[d].best_targets:axes[0].add_patch(Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor='#23394e',linewidth=2))
    axes[0].set_title('Exact value after fixing the first action')
    lookup={int(r['depth']):r for r in decisions};regrets=[int(lookup[d]['first_move_regret']) for d in depths]
    axes[1].bar(range(len(depths)),regrets,color='#21847b')
    axes[1].set_xticks(range(len(depths)),[f'H={d}' for d in depths]);axes[1].set_ylabel('Full-game optimal U minus chosen-first-action U')
    for i,value in enumerate(regrets):axes[1].text(i,value+.05,str(value),ha='center')
    axes[1].set_ylim(0,max(regrets,default=0)+.7);axes[1].set_title('First-action regret under full-game evaluation')
    fig.tight_layout();_save(fig,directory,'02_decisions_regret');plt.close(fig)
    by={(r['scope'],r['depth'],r['algorithm']):r for r in summary}
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.6))
    for ax,metric,label in zip(axes,('node_visits','leaf_evals'),('Visits including root','Leaf evaluations')):
        for a in ALGORITHMS:ax.plot(depths,[by['precomputed',d,a][metric] for d in depths],marker='o',label=LABELS[a],color=COLORS[a])
        ax.set_xticks(depths);ax.set_xlabel('H (one player action per layer)');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Same decisions; pruning changes the amount of search');fig.tight_layout();_save(fig,directory,'03_workload');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16.2,4.8))
    for ax,scope,metric,title in zip(axes,('precomputed','precomputed','end_to_end'),('search_ms','total_ms','total_ms'),
                                    ('Game search only','Game query with reused distances','Map to decision: fresh A* distances')):
        for a in ALGORITHMS:
            rows=[by[scope,d,a] for d in depths];y=[r[f'{metric}_median'] for r in rows]
            ax.plot(depths,y,marker='o',label=LABELS[a],color=COLORS[a])
            ax.fill_between(depths,[r[f'{metric}_q1'] for r in rows],[r[f'{metric}_q3'] for r in rows],color=COLORS[a],alpha=.15)
        ax.set_xticks(depths);ax.set_xlabel('H');ax.set_ylabel('Per call (ms)');ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.text(.5,.01,'One input only. Lines: batch medians; bands: Q1-Q3 of repeated batch means, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));_save(fig,directory,'04_runtime');plt.close(fig)
    chosen=[d for d in (2,4) if d in refs] or depths[-2:]
    fig,axes=plt.subplots(len(chosen),1,figsize=(12.5,2.1*len(chosen)+1),squeeze=False)
    for ax,d in zip(axes[:,0],chosen):
        rows=action_ledger(matrix,case['treasure_values'],refs[d].principal_variation)
        body=[[r[f] for f in ('step','player','target','distance','gain','score_a','score_b','utility')] for r in rows]
        ax.axis('off');table=ax.table(cellText=body,colLabels=['Step','Player','Target','Distance','Gain','A score','B score','U'],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.6)
        ax.set_title(f'H={d}: '+ ' -> '.join(refs[d].principal_variation)+(' (complete)' if d>=len(case['treasure_values']) else ' (truncated)'),pad=9,fontsize=11)
    fig.suptitle('A canonical best-response path; tied alternatives may exist',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));_save(fig,directory,'05_action_ledgers');plt.close(fig)


def analyze_results(result_dir):
    folder=Path(result_dir).resolve();config=json.loads((folder/'config.json').read_text(encoding='utf8'))
    if config.get('status')!='completed':raise ValueError('仅分析已完成的测量')
    _verify_sources(folder,config)
    path=(folder/config['input_file']).resolve()
    if not path.is_relative_to(folder) or sha256(path.read_bytes()).hexdigest()!=config['input_sha256']:
        raise ValueError('保存的输入路径或哈希错误')
    _,case=load_input(path);matrix=json.loads((folder/'distance_matrix.json').read_text(encoding='utf8'))
    refs={d:reference_game(matrix,case['treasure_values'],d) for d in config['depths']}
    rows=_read_rows(folder/'raw.csv');_validate_batches(rows,config)
    if len(rows)!=config.get('raw_rows') or sum(r['batch_loops'] for r in rows)!=config.get('total_timed_calls'):
        raise ValueError('行数或正式调用次数与配置不符')
    decisions=_validate_evidence(folder,case,matrix,refs,rows,config)
    summary,paired=summarize(rows)
    _write_csv(folder/'summary.csv',summary);_write_csv(folder/'paired_summary.csv',paired)
    result={'run_id':config['run_id'],'input_count':1,'depths':config['depths'],'raw_rows':len(rows),
            'total_timed_calls':sum(r['batch_loops'] for r in rows),'all_correct':True,
            'aggregation':'Within one input and configuration: median and linearly interpolated Q1/Q3 of batch per-call means; speedup is ratio of algorithm medians, not an independent-map statistic.',
            'summary':summary,'paired':paired,'decisions':[{k:(int(v) if k not in ('chosen_target','is_complete_game') else v) for k,v in r.items()} for r in decisions]}
    make_figures(folder,case,matrix,refs,decisions,summary)
    _write_json(folder/'summary.json',result)
    config.update(analysis_status='completed',analysis_completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    _write_json(folder/'config.json',config)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='题四距离和深度实验的核验、汇总与绘图')
    parser.add_argument('--results',type=Path,default=ROOT/'results/problem4_depths')
    args=parser.parse_args(argv)
    try:report=analyze_results(args.results)
    except (ValueError,OSError,KeyError,ImportError) as error:parser.exit(2,f'分析失败：{error}\n')
    print(json.dumps({k:report[k] for k in ('input_count','raw_rows','total_timed_calls','all_correct')},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

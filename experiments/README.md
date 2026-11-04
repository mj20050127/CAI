# 实验运行与结果索引

完整操作见 [复现指南](../docs/复现指南.md)，首次使用从 [项目首页](../README.md) 开始。这里说明10轮对照的输入、运行器、独立参考和统计口径。

## 数据与对照因素

| 轮次及正式目录 | 输入与规模 | 只改变的因素 | 独立正确性参考 |
| --- | --- | --- | --- |
| `problem1_heuristics` | 21/51/101迷宫各30张，障碍概率0.2；BFS拒绝不可达图 | h=0／曼哈顿 | BFS及路径合法性 |
| `problem2_heuristics` | 30张31×31无障碍图；权1/3/8，概率0.7/0.2/0.1 | 零／原始／安全修正启发 | 反向Dijkstra、逐格与逐有向边审计 |
| `problem2_lookahead` | 同30张加权图 | h0／一步h1／两步h2 | 最优代价、可采纳性、一致性与支配关系 |
| `problem2_lookahead_optimized` | 同30张加权图 | 旧／快速一步的等值实现 | 逐格值、路径、扩展序与计数等价 |
| `problem2_two_step_optimized` | 同30张加权图 | 四种启发或等值求值实现 | 新旧两步逐格及搜索行为等价 |
| `problem3_minimax_alpha_beta` | 三叉满树深度4/6/8各30棵；叶值整数[-50,50] | 同树同序，是否剪枝 | 独立自底向上回传 |
| `problem3_visit_order` | 同90棵随机树；每树10随机序和1真值序 | 子节点顺序及其准备成本 | 完整MM与独立字典参考 |
| `problem4_depths` | 官方7×7图与4宝藏；H=1至4 | 深度、MM／AB、矩阵复用／端到端 | Dijkstra及完整行动排列回传 |
| `problem4_order` | 同官方输入，H=4，24个固定全局优先级 | 全局访问优先级 | 48个MM／AB配置审计及独立参考 |
| `problem4_gain` | 同官方输入，H=4 | 同一递归实现的原序／动态Gain排序 | 独立完整参考与生产基线交叉核对 |

前三类随机主实验均把官方输入单列，不并入随机样本。题一主种子20261021，题二、三20261022；每个实际输入另有哈希。题四只有1张官方图，批次和优先级配置均不增加独立地图数。

## 运行器与分析器

| 结果目录 | 采集模块 | 汇总绘图模块 |
| --- | --- | --- |
| `problem1_heuristics` | `experiments.exp1_heuristic` | `experiments.analyze_maze_heuristics` |
| `problem2_heuristics` | `experiments.exp2_weighted_terrain` | `experiments.analyze_weighted_heuristics` |
| `problem2_lookahead` | `experiments.exp2_lookahead` | `experiments.analyze_lookahead` |
| `problem2_lookahead_optimized` | `experiments.exp2_lookahead_optimized` | `experiments.analyze_lookahead_optimized` |
| `problem2_two_step_optimized` | `experiments.exp2_two_step_optimized` | `experiments.analyze_two_step_optimized` |
| `problem3_minimax_alpha_beta` | `experiments.exp3_minimax_alpha_beta` | `experiments.analyze_game_tree` |
| `problem3_visit_order` | `experiments.exp3_visit_order` | `experiments.analyze_visit_order` |
| `problem4_depths` | `experiments.exp4_depths` | `experiments.analyze_treasure_depths` |
| `problem4_order` | `experiments.exp4_order` | `experiments.analyze_treasure_order` |
| `problem4_gain` | `experiments.exp4_gain` | `experiments.analyze_treasure_gain` |

测量用新的 `--out`；汇总用 `--results` 指向新测目录或整轮复制后的副本。runner默认继续分析绘图，`--no-analysis`只跳过该阶段。analyzer会改写目标目录的派生结果和配置时间，不是只读操作。完整可复制命令见复现指南。

## 计时与统计

- 正式大多5批，Gain10批；先预热与校准，同输入中的方法使用同一批内调用数，并交替或均衡顺序。实际参数和执行次序在config、calibration及raw中。
- A*有效出队含起终点，堆峰值为物理堆条目，不是进程内存；博弈访问含根与叶。`pruned_branches`是跳过的直接后继分支数，不是省去子树的全部节点数。
- search计内部搜索，A*包含启发求值，Gain包含递归内的动态排序；题三复制、真值预计算与预排序另计准备阶段，纳入完整total。接口时间还包括输入准备或校验，具体边界以各轮分析为准。
- 先在每输入内汇总重复批次，再跨独立输入汇总。节省率与倍率先同输入配对计算。随机10序先在每棵树内汇总；Q1至Q3表示分散，不是置信区间。
- 正式性能计时关闭trace；读取文件、独立参考、正确性审计和绘图在计时外。题四端到端从已在内存中的地图开始，不包含磁盘读取。
- 真值排序事先求完整子树值，它不是免费的实用启发；动态Gain的求值与排序计入搜索和总时。更少节点并不保证完整流程更快。

## 保存的证据

| 文件或目录 | 用途 |
| --- | --- |
| `config.json` | 参数、环境、输入及源码哈希、运行顺序、采集和分析状态 |
| `raw.csv` | 每个配置和批次的原始计时、计数与正确性记录 |
| `calibration.csv` | 校准和实际循环次数 |
| `dataset/` 或 `input/` | 实际地图/树/官方输入及manifest；基础轮复用模式还可能引用源dataset |
| `summary*`、`paired*`或题四专用analysis文件 | 汇总、配对效果及分散 |
| `paths*`、`proof*`、`equivalence*`、`heuristic_audit*`、轨迹等 | 路径、代价、性质、等价与剪枝证据，按轮次保存 |
| `recipes.jsonl.gz` | 题三实际访问排列，用于重建顺序 |
| `source_baseline.zip`及已有`source_measurement.zip` | 对应轮次运行/分析时的源码快照 |
| `figures/` | 正式PNG/SVG原图 |
| 各轮中文分析Markdown | 公式、设置、结果、原因和适用范围 |

保留整轮目录与原始哈希，避免只留下截图。README修改不影响源码快照校验；源码变化后，按对应轮次快照核验旧结果。

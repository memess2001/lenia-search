# Project 24: Unified Life Search — 项目状态

> **最后更新**：2026-03-24
> **项目目录**：`~/Projects/project_lenia_search/`
> **完整实施计划**：`docs/IMPLEMENTATION_PLAN.md`

---

## 一、项目是什么

### 一句话

用两条搜索路径——人类设计的 Lenia 函数族（Track A）和 LLM 自由生成的数学表达式（Track B）——在同一个 MAP-Elites archive 里竞争，回答 **"什么数学能产生生命？生命有多少种数学实现？"**

### 核心卖点（论文灵魂）

**不是**"用了 LLM 搜 ALife"（ASAL 已做过）。**是**：

1. **统一竞争 archive**——Lenia 家族和 LLM 自由表达式在同一个行为空间里 PK
2. **跨函数族搜索**——15 种核函数 × 8 种增长函数 × 4 种几何，不只是调参
3. **零重叠发现**——LLM 找到了 Track A 25000+ 变种从没覆盖过的行为区域

### 投稿方向

- ALIFE 2027 或 Artificial Life (MIT Press)
- 可能 Nature Machine Intelligence（如果数据足够丰富）

---

## 二、已完成的工作

### Track A: MAP-Elites 参数化搜索

| 阶段 | 搜索空间 | 变种数 | Archive bins | 状态 |
|------|---------|--------|-------------|------|
| Phase 1a | 4核 × 4增长 × flat | 25,350 | 282/400 (70.5%) | ✅ 饱和 |
| Phase 1b 扩展函数 | 15核 × 8增长 × flat | +13,260 | 282/400 (不变) | ✅ 新函数没打开新区域 |
| Phase 1b 几何 | 15核 × 8增长 × 4几何 | +12,000 | 287/400 (71.8%) | ✅ 几何打开了新区域 |

**关键发现**：
- power_law 和 yukawa 核函数产生最多 archive cells（长程力 = 多样性）
- 所有 15 种核函数都在 archive 里出现（没有"死"函数族）
- 双曲面上存活了同参数下平面/球面全死的 entity → **几何塑造生命**
- Track A 本地搜索已饱和，需上云 GPU 做更大规模搜索

### Track B: LLM-Guided 自由表达式搜索

| Round | 模式 | 迭代次数 | 插入 | 独占 bins | 状态 |
|-------|------|---------|------|----------|------|
| Round 1 | 单场 | 500 | 3 次操作 → 2 bins | 2 | ✅ |
| Round 2 | 单场 | 500 | 5 次操作 → 3 新 bins | 5 (累计) | ✅ |
| Round 3 | 单场 | 1000 | 3 次操作 → 0 新 bins | 5 (不变) | ✅ |
| **Round 4** | **双场** | 1000 | 进行中 | ? | 🟢 运行中 |

**关键发现**：
- Track B 的 5 个 entry 和 Track A 的 282 个 entry **零重叠**
- Track B 最高 fitness (1.007) > Track A 最高 (1.005)
- Track B 平均 alive_fraction (0.804) > Track A (0.598)
- Track B 占据了行为空间的"右上角"（高复杂度 + 高存活率）
- LLM 使用了 `cos(4π * laplacian(A))`、`tanh(2 * laplacian(A)²)` 等**标准 Lenia 无法表达的数学结构**

### 基础设施

| 组件 | 文件 | 功能 |
|------|------|------|
| Lenia 模拟器 | `src/lenia_core.py` | 15 种核函数 + 8 种增长函数，FFT 卷积 |
| 曲面几何 | `src/geometric_lenia.py` | 球面、环面、双曲面 Lenia |
| 几何工厂 | `src/geometry_factory.py` | 自动分派到正确的模拟器 |
| 基因组 | `src/genome.py` | 搜索空间定义 + 随机/变异 |
| 评估器 | `src/evaluator.py` | 7 种度量（alive, complexity, entropy, clusters...） |
| MAP-Elites | `src/map_elites.py` | Track A 搜索循环 |
| 表达式编译器 | `src/expression_compiler.py` | 数学字符串 → PyTorch 函数 + 双场支持 |
| LLM 突变器 | `src/llm_mutator.py` | Claude Sonnet API + 单场/双场 prompt |
| LLM 搜索 | `src/llm_search.py` | Track B 搜索循环 + 单场/双场模式 |
| Track 对比 | `analyze_tracks.py` | A vs B 覆盖率分析 |
| 可视化 | `visualize.py` | Archive heatmap + GIF |

---

## 三、核心数据

### Archive 最终状态（截至 2026-03-24）

```
总 entries:    287/400 bins (71.8%)
Track A:       282 bins
Track B:       5 bins (独占，零重叠)
总 evaluated:  ~26,000 (Track A) + ~2,000 (Track B)
```

### Track A vs Track B 对比

| 指标 | Track A (MAP-Elites) | Track B (LLM) |
|------|---------------------|---------------|
| Entries | 282 | 5 |
| 独占 bins | 282 | 5 |
| 重叠 | 0 | 0 |
| 平均 fitness | 0.436 | **0.609** |
| 最高 fitness | 1.005 | **1.007** |
| 平均 alive | 0.598 | **0.804** |

### Track B 发现的 Top 5 生命形态

| # | bin | fitness | alive | clusters | 数学特征 |
|---|-----|---------|-------|----------|---------|
| 1 | [19,19] | 1.007 | 0.976 | 10 | cos(4π·laplacian) 调制 |
| 2 | [18,19] | 0.949 | 0.990 | 1 | tanh(laplacian²) 耦合 |
| 3 | [18,18] | 0.947 | 0.943 | 14 | 多操作 max + 二阶导 |
| 4 | [1,18] | 0.092 | 0.936 | 6 | 双峰 growth + tanh |
| 5 | [0,3] | 0.050 | 0.176 | 1 | A·(1-A) logistic 项 |

---

## 四、还要做什么

### 短期（本周-下周）

1. **Track B Round 4 双场** — 1000 迭代进行中，看双场能否打开单场到不了的区域
2. **Project 10 Phase 2** — 等跑完 (945 个实验点)，画新的 scaling curves
3. **Round 5：复数场** — 先固定模、只让相位演化（另一个 Claude 的建议）
4. **Round 6：激进 prompt** — 加入手性、守恒律、开放边界等方向

### 中期（Phase 2，第 5-7 周）

5. **Track A 扩展到 35 核 × 18 增长 × 12 几何** — 需要上云 GPU
6. **云 GPU 密集搜索** — 预算 $300-500，Vast.ai 或 Lambda Labs
7. **Track B 5000+ 迭代** — 更多单场 + 双场 + 复数场
8. **统一竞争实验** — 大规模对比 Track A vs Track B 的覆盖率

### 长期（Phase 3-4，第 8-14 周）

9. **分析 + 分类** — 对所有发现的生命形态做分类学
10. **论文写作** — 核心图表：Track A vs B archive 对比、生命 gallery、几何对比
11. **开源发布** — `geometric-lenia` 库

---

## 五、外部评审意见摘要

### Gemini 的核心意见
- 时间线应该是 3-4 个月，不是 1 个月
- **不要拆成两篇论文**——合在一起才有冲击力
- 别 claim "first LLM for ALife"（ASAL 已存在）
- 曲面 Lenia 是"甜点"不是"主菜"
- 统一评估层是地基

### GPT 的核心意见
- 值得做，更适合作为"研究平台/系列论文"
- 和 Gemini 一致：核心卖点 = 统一竞争基准

### 另一个 Claude 的维度建议（按优先级）
1. ⭐ 手性规则（破缺宇称对称）— 实现成本极低，ALife 叙事强
2. ⭐ 质量+动量守恒 — 推向 Navier-Stokes 方向
3. ⭐ 开放系统边界 — "代谢" = autopoiesis
4. 复数场 — 先固定模只让相位演化
5. 元规则/哥德尔自指 — 体量太大，不赶 deadline

---

## 六、文件结构

```
~/Projects/project_lenia_search/
├── src/
│   ├── __init__.py
│   ├── lenia_core.py           ← 15 核 + 8 增长 Lenia 模拟器
│   ├── geometric_lenia.py      ← 球面/环面/双曲面 Lenia
│   ├── geometry_factory.py     ← 几何分派工厂
│   ├── genome.py               ← 搜索空间定义
│   ├── evaluator.py            ← 统一评估 (7 度量)
│   ├── batch_lenia.py          ← 批量运行器
│   ├── map_elites.py           ← Track A MAP-Elites
│   ├── expression_compiler.py  ← 单场 + 双场表达式编译
│   ├── llm_mutator.py          ← Claude API 突变器 (单场/双场)
│   └── llm_search.py           ← Track B LLM 搜索循环
├── run_search.py               ← Track A 入口
├── run_llm_search.py           ← Track B 入口 (--mode single|dual)
├── analyze_tracks.py           ← Track A vs B 对比分析
├── visualize.py                ← Archive heatmap + GIF
├── docs/
│   ├── IMPLEMENTATION_PLAN.md  ← 完整 14 周实施计划
│   └── PROJECT_STATUS.md       ← THIS FILE
├── results/
│   ├── archive.json            ← 统一 archive (287 entries)
│   ├── archive_phase1a_4x4.json     ← Phase 1a 备份
│   ├── archive_phase1b_flat_only.json ← Phase 1b 纯平面备份
│   ├── track_b_gallery.png     ← Track B 生命可视化
│   ├── map_elites_heatmap.png
│   └── *.txt                   ← 各轮搜索日志
└── README.md
```

---

## 七、关键命令参考

```bash
# Track A: MAP-Elites 搜索 (从 archive 继续)
python run_search.py --hours 8 --batch_size 8 --resolution 64 --steps 2000 --archive results/archive.json

# Track B: LLM 单场搜索
ANTHROPIC_API_KEY="sk-..." python run_llm_search.py --iterations 1000 --archive results/archive.json

# Track B: LLM 双场搜索
ANTHROPIC_API_KEY="sk-..." python run_llm_search.py --iterations 1000 --mode dual --archive results/archive.json

# Track A vs B 对比分析
python analyze_tracks.py --archive results/archive.json --detailed

# 可视化 archive heatmap
python visualize.py
```

---

## 八、API 费用追踪

| Round | 迭代 | 模型 | 估计费用 |
|-------|------|------|---------|
| Round 1 | 500 | Claude Sonnet | ~$1.50 |
| Round 2 | 500 | Claude Sonnet | ~$1.50 |
| Round 3 | 1000 | Claude Sonnet | ~$3.00 |
| Round 4 (进行中) | 1000 | Claude Sonnet | ~$3.00 |
| **累计** | **3000** | | **~$9.00** |

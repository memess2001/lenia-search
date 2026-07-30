# Project 24: Unified Life Search — 完整实施计划

> **项目定位**：旗舰级研究项目（3-4 个月），不是短平快工作。
> **核心问题**：什么数学能产生生命？生命有多少种数学实现？
> **论文目标**：ALIFE 2026/2027 或 Nature Machine Intelligence
> **最终更新**：2026-03-19

---

## 0. 战略定位（基于 Gemini/GPT 评审反馈）

### 核心卖点（三选一都不够，缺一不可）

1. **跨函数族搜索**：~35 种核函数族 × ~35 种空间几何 × ~18 种增长函数 × 通道/时间结构。不是在固定框架内调参（Leniabreeder/IMGEP 做的事），而是在函数族之间搜结构。
2. **LLM-guided 自由算式生成（Track B）**：FunSearch/LaSR 式方法首次用于"生命方程发现"。不预设任何框架，让 LLM 在表达式树空间中搜索能产生生命的数学。
3. **统一竞争 Archive**：两条路径的发现在同一个 MAP-Elites archive 中 PK。核心产出 = "Lenia 家族覆盖了生命空间的 X%"——这个数字回答的是"生命的多重可实现性程度"。

### 不要做的事

- ❌ 不 claim "first LLM-guided GP for ALife"（ASAL 已存在）
- ❌ 不拆成两篇论文（拆开后每篇都是 incremental，合在一起才有冲击力）
- ❌ 不把曲面 Lenia 当主菜（它是甜点，核心是"什么数学产生生命"）

### 正确的 Framing

> "We construct the first unified competitive benchmark for artificial life discovery, where human-designed mathematical frameworks (the Lenia family across 35 kernel types and 35 geometries) compete directly against LLM-discovered free-form update rules in a shared MAP-Elites archive. We find that Lenia covers only X% of the viable life space, and LLM-guided search discovers Y fundamentally new mathematical structures that produce life."

---

## 1. 当前进度

### 已完成

| 组件 | 状态 | 位置 |
|------|------|------|
| 基础 Lenia 模拟器 | ✅ | `src/lenia_core.py` |
| 基因组定义 + 变异 | ✅ (4 核 × 4 增长) | `src/genome.py` |
| 统一评估层 | ✅ (7 指标) | `src/evaluator.py` |
| 批量运行器 | ✅ | `src/batch_lenia.py` |
| MAP-Elites 搜索 | ✅ | `src/map_elites.py` |
| 可视化 | ✅ | `visualize.py` |
| Phase 1 初筛 | ✅ 25,350 变种，274/400 bins (68.5%) | `results/archive.json` |

### 未完成

| 组件 | 状态 | 优先级 |
|------|------|--------|
| 扩展核函数族（4 → 35） | ❌ | P0 |
| 扩展增长函数（4 → 18） | ❌ | P0 |
| 扩展空间几何（1 → 35） | ❌ | P1 |
| LLM-guided Track B | ❌ | P2（核心差异化） |
| 统一竞争实验 | ❌ | P3 |
| 云 GPU 部署 | ❌ | P3 |
| 论文 | ❌ | P4 |

---

## 2. 完整搜索空间规格

### 2.1 核函数族（目标 ~35 种）

#### 第一批扩展（P0，立即实现，+11 种 → 共 15 种）

| # | 类型 | 数学形式 | 物理直觉 |
|---|------|---------|---------|
| 1 | gaussian | exp(-((r-μ)/σ)²) | 标准 Lenia |
| 2 | polynomial | (4r(1-r))^α | 有限支撑 |
| 3 | mexican_hat | (1-r²)·exp(-r²/2) | 侧抑制（神经科学） |
| 4 | bump | exp(-1/(1-r²)) | C∞ 紧致支撑 |
| **5** | **cosine** | cos(πr/2)^α | 紧致平滑 |
| **6** | **bessel** | J₀(αr) | 自然界的波 |
| **7** | **sinc** | sin(πr)/(πr) | 理想低通滤波 |
| **8** | **gabor** | exp(-r²/2σ²)·cos(ωr) | 视觉皮层模型 |
| **9** | **step_ring** | 环形阶梯函数 | 最接近离散 CA |
| **10** | **power_law** | r^(-α) with cutoff | 长程相互作用 |
| **11** | **yukawa** | exp(-αr)/r | 核物理短程力 |
| **12** | **elliptical** | 椭圆形核 | 方向偏好 |
| **13** | **spiral** | r·exp(iθ·n) 的模 | 手性核 |
| **14** | **fourier_series** | Σ(aₙcos(nr) + bₙsin(nr)) | 傅里叶展开 |
| **15** | **rbf_mixture** | Σwᵢ·exp(-((r-cᵢ)/σᵢ)²) | 高斯混合 |

#### 第二批扩展（Phase 2，+20 种 → 共 35 种）

- 振荡族：airy, fractal_sine
- 奇异族：cantor, devil_staircase, weierstrass
- 长程族：coulomb_screened, log_decay
- 各向异性族：dipolar
- 可学习族：learned_mlp, b_spline, wavelet
- 特殊函数族：legendre, chebyshev, laguerre, hermite_gauss, zernike, spherical_harmonic

### 2.2 增长函数族（目标 ~18 种）

#### 第一批（P0，+4 种 → 共 8 种）

| # | 类型 | 特性 |
|---|------|------|
| 1 | gaussian_bell | 标准 Lenia |
| 2 | step_function | 最接近 Conway |
| 3 | asymmetric_bell | 方向偏好 |
| 4 | bistable | 两种物态 |
| **5** | **laplace_peak** | 尖峰，对密度更敏感 |
| **6** | **cauchy_peak** | 重尾 |
| **7** | **sigmoid_pair** | 带通 |
| **8** | **relu_like** | 分段线性 |

#### 第二批（Phase 2，+10 种 → 共 18 种）

- 多稳态：tristable, periodic
- 极端行为：absolute_inhibition, ramp
- 动态/记忆：hysteresis, history_dependent, fatigue

### 2.3 空间几何（目标 ~35 种）

#### 第一批（P1，+3 种 → 共 4 种）

| # | 几何 | 实现方式 | 特征 |
|---|------|---------|------|
| 1 | **flat_plane** | 标准 FFT + 周期边界 | 基线 |
| **2** | **sphere_S2** | HEALPix + 球谐变换 (healpy) | 正曲率，有限无边 |
| **3** | **torus_embedded** | 嵌入 R³ 的环面 | 内侧负曲率，外侧正曲率 |
| **4** | **hyperbolic_H2** | Poincaré 圆盘模型 | 负曲率 |

#### 第二批（Phase 2，+8 种 → 共 12 种）

- Klein bottle, projective plane, double torus
- Gaussian bump, saddle landscape, random curvature, curvature gradient
- cone

#### 第三批（Phase 3，+23 种 → 共 35 种）

- 奇异空间：cusp, branched surface, fractal surface
- 高维空间：S3 slice, flat 3D
- 乘积空间：S1×R, S2×S1, H2×R
- 动态几何：expanding, contracting, breathing, entity_driven, curvature_from_state
- 离散拓扑：random graph, small world, scale free, hypergraph

### 2.4 其他搜索维度

| 维度 | Phase 0 (当前) | Phase 2 目标 |
|------|---------------|-------------|
| 通道数 | 1 | 1, 2, 3, 4 |
| 通道耦合 | none | linear, inhibitory, predator_prey |
| dt | 0.1 | 0.1, 0.05, 0.02 |
| 记忆深度 | 0 | 0, 1, 3 |
| 自修改 | none | none, flow_lenia, local_mutation |

---

## 3. 双轨搜索架构

### Track A: Lenia 家族 MAP-Elites（已部分实现）

```
基因组空间 → 随机采样/变异 → BatchLenia 模拟 → 统一评估 → Archive 更新
```

- 当前：4 核 × 4 增长 × 1 几何 = 16 种组合
- Phase 1b 目标：15 核 × 8 增长 × 4 几何 = 480 种组合
- Phase 2 目标：35 核 × 18 增长 × 12 几何 × 通道/时间 = 数百万种组合 → MAP-Elites 智能采样

### Track B: LLM-Guided 自由算式搜索（待实现）

```
LLM 生成表达式 → 编译为 update_rule → 统一模拟器运行 → 统一评估 → 同一个 Archive
```

核心组件：
1. **表达式编译器**：将 LLM 输出的数学表达式编译为 PyTorch 可执行函数
2. **LLM 突变器**：给定两个高分"父代"表达式，LLM 生成新变体（FunSearch 式）
3. **概念抽象器**：从高分表达式中提取可复用结构（LaSR 式）
4. **安全沙箱**：LLM 生成的代码可能有 bug，需要 try/except + 超时保护

LLM 选择：Claude API（Sonnet，~$0.003/次突变）

### 统一 Archive

- 行为空间：complexity × alive_fraction（2D，20×20 bins = 400 cells）
  - 可扩展为 3D：+ num_clusters
- Fitness：complexity（更复杂 = 更好的生命）
- 每个 cell 存储：(genome_or_expression, fitness, full_metrics, source_track)
- `source_track` 标记来自 Track A 还是 Track B → 这是最终论文的核心数据

---

## 4. 实施时间线（修正版，3-4 个月）

### Phase 1a: 基座验证 ✅ 完成（第 1 周）

- [x] 基础 Lenia 模拟器
- [x] MAP-Elites pipeline
- [x] 统一评估层
- [x] 初筛 25,350 变种（4 核 × 4 增长 × flat）

### Phase 1b: 扩展搜索空间（第 2-3 周）

- [ ] 扩展核函数族：4 → 15 种
- [ ] 扩展增长函数：4 → 8 种
- [ ] 实现球面几何（healpy + 球谐卷积）
- [ ] 实现环面 + 双曲面几何
- [ ] M1 Pro 上重新搜索：15 核 × 8 增长 × 4 几何
- [ ] 分析初筛结果：哪些函数族/几何产生了新类型的生命？

**里程碑**：在 4 种几何上都观察到存活的 Lenia entity，archive 覆盖率反映几何对生命形态的影响。

### Phase 2a: Track B 搭建（第 4-5 周）

- [ ] 实现表达式编译器（数学字符串 → PyTorch 函数）
- [ ] 实现 LLM 突变器（Claude API）
- [ ] 实现概念抽象器
- [ ] 安全沙箱（超时、异常捕获）
- [ ] Track B 单独验证：LLM 能否发现至少一个存活的 entity？

**里程碑**：Track B 成功往 archive 写入至少 10 个存活的生命形态。

### Phase 2b: 双轨竞争（第 5-7 周）

- [ ] Track A + Track B 同时运行，竞争同一个 archive
- [ ] 扩展核函数到 35 种（第二批）
- [ ] 扩展增长函数到 18 种（第二批）
- [ ] 扩展几何到 12 种（第二批）
- [ ] 加入多通道（2-4 通道）
- [ ] M1 Pro 夜间连续搜索

**里程碑**：archive 中同时有 Track A 和 Track B 的发现，可以开始比较覆盖率。

### Phase 3: 云 GPU 密集搜索（第 8-10 周）

- [ ] 部署到云 A100（Vast.ai / Lambda Labs）
- [ ] Track A 大规模搜索：50,000-100,000 变种
- [ ] Track B 大规模搜索：5,000-10,000 LLM 生成的表达式
- [ ] 预算：$300-500
- [ ] 扩展几何到完整 35 种（第三批，含动态几何）

**里程碑**：
- Archive 总覆盖率 > 50%
- Track A 覆盖了 X% 的 bins
- Track B 覆盖了 Y% 的 bins
- 其中 Z% 只有 Track B 能到达（Lenia 框架无法覆盖的区域）

### Phase 4: 分析 + 论文（第 11-14 周）

- [ ] 核心图表：Track A vs Track B 的 archive 覆盖对比图
- [ ] 分类学：对发现的生命形态进行分类
- [ ] 找最有趣的 top 50，生成动画 GIF
- [ ] 数学分析：Track B 发现的新生命用了什么数学结构？
- [ ] 论文撰写

**里程碑**：论文初稿完成，投 ALIFE 或 Nature Machine Intelligence。

---

## 5. 评估指标详细设计

### MAP-Elites 行为空间

**主方案（2D）**：
- X 轴：complexity（LZMA 压缩比，0-1）
- Y 轴：alive_fraction（存活比例，0-1）
- 20 × 20 bins = 400 cells

**备选方案（3D）**：
- + Z 轴：num_clusters（entity 数量，1-20）
- 20 × 20 × 10 = 4,000 cells

### Fitness 函数

```
fitness = complexity × longevity_factor
```

其中 longevity_factor = min(longevity / 3000, 1.0)

理由：既要复杂又要持久。纯复杂度可能奖励短暂的爆炸性 pattern。

### "死亡"判定

- alive_fraction < 0.001 持续 200 步以上 → dead
- 或者 mass 完全不变持续 500 步以上 → frozen（归为 dead）

### "有趣"分级

| 等级 | 条件 | 意义 |
|------|------|------|
| ★ | alive & complexity > 0.3 | 基本存活 |
| ★★ | + num_clusters > 1 | 有独立 entity |
| ★★★ | + 运动（质心位移 > 5 pixels/1000 steps）| 能移动 |
| ★★★★ | + 自修复（扰动后恢复）| 鲁棒生命 |
| ★★★★★ | + 交互（多 entity 的行为受彼此影响）| 生态 |

### 额外度量（Phase 2+）

- **运动性**：质心位移速度
- **自修复**：随机删除 10% 的 mass 后 500 步内的恢复比例
- **对称性**：旋转/反射对称度
- **Persistent homology**：Betti 数作为形态拓扑特征

---

## 6. 技术依赖

### 已安装

| 库 | 用途 |
|----|------|
| PyTorch | FFT 卷积、张量运算 |
| NumPy / SciPy | 数值计算、connected components |
| matplotlib | 可视化 |
| tqdm | 进度条 |

### 需要安装

| 库 | 用途 | 阶段 |
|----|------|------|
| healpy | 球面几何（HEALPix + 球谐变换） | Phase 1b |
| trimesh | 通用曲面网格（环面、双曲面） | Phase 1b |
| anthropic | Claude API（Track B 的 LLM 调用） | Phase 2a |
| Pillow / imageio | GIF 动画生成 | 已有 |
| giotto-tda | Persistent homology（Phase 3 分析） | Phase 3 |

### 硬件

| 阶段 | 硬件 | 每晚可跑变种数 | 成本 |
|------|------|---------------|------|
| Phase 1-2 | M1 Pro (CPU) | ~3,000-5,000 | 已有 |
| Phase 3 | 云 A100 (1 张) | ~20,000-50,000 | $8-16/晚 |
| Phase 3 | 云 A100 (4 张) | ~80,000-200,000 | $32-64/晚 |
| Phase 3 总预算 | | | **$300-500** |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| Track B 生成的表达式大部分无法编译 | LLM 搜索效率低 | 严格的 prompt 模板 + few-shot examples + 语法校验 |
| 球面/双曲面上 Lenia 全部死亡 | 无法证明"几何塑造生命" | 先在球面上手动调参找到存活种子，再用 MAP-Elites 扩展 |
| Track A 和 Track B 覆盖完全重叠 | 没有"竞争"的故事 | 这本身也是结果——"Lenia 框架已经覆盖了所有可能" |
| ASAL 后续工作抢先发表类似结果 | 新颖性削弱 | 我们的差异化在"统一竞争基准"和"多重可实现性"哲学定位 |
| 云 GPU 预算超支 | 实验不完整 | 先在 M1 Pro 充分初筛，只把最有价值的搜索空间区域交给云 |

---

## 8. 论文骨架（预览）

**标题候选**：
- "What Mathematics Makes Life? A Unified Search Across Function Spaces and Geometries"
- "The Multiple Realizability of Artificial Life: Lenia vs. LLM-Discovered Dynamics"

**核心图表**（论文的灵魂）：
- **Figure 1**: 统一 Archive 热力图，左半 = Track A (Lenia)，右半 = Track B (LLM)，颜色 = fitness
- **Figure 2**: 覆盖率柱状图——Track A 覆盖了 X% bins，Track B 覆盖了 Y%，重叠 Z%
- **Figure 3**: 最有趣的 top 20 生命形态 gallery（动画截图 + 数学公式）
- **Figure 4**: 不同几何上的 entity 形态对比（flat vs sphere vs hyperbolic）
- **Figure 5**: Track B 发现的全新数学结构——LLM 找到了什么人类没想到的东西？

---

## 9. 与其他项目的关系

| 项目 | 如何受益于 Project 24 |
|------|---------------------|
| **#9 生命带** | 直接对 archive 做压缩距离分析 → 几乎免费的额外论文 |
| **#4 元认知几何** | 24 发现的 entity 有元认知吗？ |
| **#5 拓扑相变** | 不同几何上的集体行为有不同拓扑相变？ |
| **#22 感质色谱** | 曲面几何作为"生存压力"影响内部表征？ |
| **#23 多重可实现性** | 24 的核心问题就是生命的多重可实现性 |

---

## 10. 立即下一步（本周）

1. **扩展 `genome.py` 和 `lenia_core.py`**：加入第一批新核函数（+11 种）和增长函数（+4 种）
2. **实现球面几何**：安装 healpy，写 `geometric_lenia.py`
3. **重启 MAP-Elites 搜索**：用扩展后的 15×8×4 搜索空间
4. **分析 Phase 1a 结果**：从已有的 274 个 archive cells 中找最有趣的变种

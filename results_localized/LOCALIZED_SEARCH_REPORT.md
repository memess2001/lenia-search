# Track A Localized Creature Search — 实验报告

**日期**: 2026-03-29
**运行时间**: 00:32 → 08:33 (8 hours, CPU)
**目的**: 测试参数化 Lenia 框架能否通过 localized fitness + center blob 初始化找到 localized creature

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 分辨率 | 128×128 |
| 步数/评估 | 8,000 |
| Batch size | 10 |
| 初始化 | mixed (50% center_blob + 50% random_quarter) |
| Fitness | `complexity × alive × (1-alive) × 4` |
| 搜索空间 | 15 kernels × 8 growth × 4 geometries |
| 设备 | CPU (M1 Pro) |

**Fitness 设计意图**: `a×(1-a)` 在 alive=0.5 时最大, 希望奖励"部分存活"的结构而非全死/全满。乘以 complexity 避免简单纹理。

---

## 2. 最终结果

| 指标 | 值 |
|------|-----|
| 总评估 | 480 genomes |
| Archive 覆盖 | **138/400 bins (34.5%)** |
| Fitness 范围 | 0.0000 — 0.7047 |
| Alive 范围 | 0.048 — 1.000 |
| Cluster 范围 | 1 — 7,835 |

### Alive fraction 分布

| 区间 | Entry 数 |
|------|---------|
| 0.0-0.1 | 4 |
| 0.1-0.2 | 8 |
| 0.2-0.3 | 16 |
| 0.3-0.4 | 17 |
| 0.4-0.5 | 19 |
| 0.5-0.6 | 20 ← fitness 峰值区 |
| 0.6-0.7 | 13 |
| 0.7-0.8 | 14 |
| 0.8-0.9 | 11 |
| 0.9-1.0 | 16 |

分布相对均匀地覆盖了行为空间，但 **fitness 最高的 entry 集中在 alive=0.4-0.7**，符合 `a*(1-a)` 的预期。

---

## 3. Top 10 by Fitness

| Rank | Fitness | Alive | Clusters | Kernel | Growth | Geometry |
|------|---------|-------|----------|--------|--------|----------|
| 1 | 0.7047 | 0.635 | 522 | yukawa | sigmoid_pair | flat_plane |
| 2 | 0.6916 | 0.577 | 5,476 | yukawa | sigmoid_pair | flat_plane |
| 3 | 0.6596 | 0.515 | 981 | yukawa | bistable | torus |
| 4 | 0.6492 | 0.656 | 2,911 | power_law | relu_like | torus |
| 5 | 0.6369 | 0.576 | 49 | cosine | relu_like | sphere |
| 6 | 0.6279 | 0.634 | 11 | cosine | relu_like | flat_plane |
| 7 | 0.6080 | 0.535 | 1,725 | power_law | relu_like | hyperbolic |
| 8 | 0.5865 | 0.641 | 12 | cosine | sigmoid_pair | hyperbolic |
| 9 | 0.5725 | 0.556 | 133 | bump | bistable | hyperbolic |
| 10 | 0.5462 | 0.448 | 134 | cosine | cauchy_peak | flat_plane |

**所有 Top 10 均为 extended pattern**：alive 0.45-0.66，大多数有数百到数千个 cluster，是全场碎片化纹理。

---

## 4. Localized Candidates 分析

仅 4 个 entry 满足 alive < 0.3 且 clusters ≤ 5：

| ID | Fitness | Alive | Clusters | Kernel+Growth+Geometry | 视觉描述 |
|----|---------|-------|----------|------------------------|----------|
| local1 | 0.1683 | 0.236 | 2 | power_law+step+flat | 两块扩散的三角形闪烁区域，不断增大 |
| local2 | 0.0330 | 0.294 | 1 | power_law+asym_bell+hyperbolic | 双曲圆盘上的均匀晶格点阵，静态 |
| local3 | 0.0086 | 0.048 | 1 | yukawa+step+sphere | 球面上微弱噪声纹理，几乎不可见 |
| local4 | 0.0060 | 0.056 | 1 | yukawa+step+torus | 环面上大面积扩散结构 |

**结论：无一是真正的 localized creature。**
- local1 虽然只有 2 个 cluster，但那是两块大面积区域，不是紧凑移动体
- local2 是规则晶格
- local3/4 是低密度噪声/扩散

时间演化截图见 `all_localized_candidates_snapshots.png`（6 帧: t=0, 200, 500, 1000, 1500, 2000）

---

## 5. 失败原因分析

### 5.1 Fitness 公式根本缺陷

`fitness = complexity × alive × (1-alive) × 4`

- 在 alive=0.5 时最大化 → 鼓励**半满的全场纹理**
- 真正的 localized creature 在 128×128 上 alive 可能只有 0.01-0.05
- 在这个 fitness 下，localized structure 的 fitness 上限极低（alive=0.05 时 `0.05×0.95=0.0475`，即使 complexity=1 也只有 0.19）

### 5.2 Center blob 初始化无效

即使从中心小 blob 开始：
- 大部分成功的 pattern 仍然扩散到全场
- Step function / relu_like 等 growth 函数产生的动力学倾向于空间扩张
- 仅有 12 个 entry 的 alive < 0.2，说明"不扩散"是少数情况

### 5.3 与 Track B 失败的对比

| | Track A Localized | Track B Localized |
|---|---|---|
| 方法 | 参数化 + blob init + localized fitness | LLM 表达式 + blob init |
| 评估数 | 480 | ~50 |
| 结果 | 138 bins, 无 creature | 3 inserts, 全 ocean/dead |
| 失败模式 | 扩散为全场纹理 | 扩散或死亡 |

**两条 Track 都失败，但原因互补**：
- Track A 有 localization 归纳偏置（finite kernel radius），但 fitness 不选择 localized
- Track B 有 fitness 但缺乏 localization 归纳偏置

---

## 6. 如果要继续尝试 Localized Search 的建议

### 方案 A：改进 fitness（推荐）
```
spatial_concentration = 1 - (mean_distance_to_centroid / max_possible_distance)
fitness = complexity × spatial_concentration × alive^0.3
```
- 直接度量活跃像素的空间集中度
- alive^0.3 对低 alive 更宽容

### 方案 B：从已知 creature 参数反向搜索
- 用经典 Lenia creature (Orbium 等) 的已知参数作为种子
- 在其邻域做局部变异搜索
- 问题：不够"discovery"，更像 parameter sweep

### 方案 C：加硬约束
- alive_fraction < 0.15 作为存活过滤
- num_clusters ∈ [1, 5] 作为存活过滤
- 在过滤后的群体中按 complexity 排序

---

## 7. 对论文定位的影响

这个实验结果进一步支持顾问的建议：

> **论文应定位为"extended self-organizing patterns across expanded function spaces"，
> 而非 localized creature discovery。**

实验证据：
- 480 次评估 × 8000 步 = 384 万步模拟，专门搜索 localized structure，失败
- Track A + Track B 两条路线、参数化 + 开放表达式两种框架，均未产生 localized creature
- 这本身是一个有价值的发现：**Lenia 扩展函数空间中 localized structure 的稀有性**

---

*报告生成时间: 2026-03-29 08:45*
*截图文件: results_localized/gifs/all_localized_candidates_snapshots.png*
*GIF 文件: results_localized/gifs/local{1-4}_256x256.gif*

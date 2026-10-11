# 【AAAI‘2023】Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton

## Metadata

- Title: Where Will Players Move Next? Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Badminton
- Authors: Kai-Shiang Chang, Wei-Yao Wang, Wen-Chih Peng
- Venue / Year: The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23), 2023
- Note Name: 【AAAI‘2023】dymf-movement-forecasting
- Paper: 本地 Zotero PDF：`C:\Users\97445\Zotero\storage\MJV3WTFS\Chang 等 - 2023 - Where Will Players Move Next Dynamic Graphs and Hierarchical Fusion for Movement Forecasting in Bad.pdf`
- Code: [CoachAI Projects / Movement Forecasting](https://github.com/wywyWang/CoachAI-Projects/tree/main/Movement%20Forecasting)
- Dataset / Artifact: 论文使用 Wang et al. (2022b) 的羽毛球单打数据；清洗后含 75 场比赛、180 局、4,325 个回合、43,191 次击球、31 名高水平运动员。论文未给出独立数据下载链接。
- Scope / Subfield: 运动轨迹预测 / 图时序建模 / 羽毛球战术分析
- Tags: `Movement Forecasting`, `Sports Analytics`, `Dynamic Graph`, `R-GCN`, `Trajectory Prediction`, `Badminton`, `Multimodal Forecasting`
- Status: DONE

## TL;DR

本文首次把羽毛球中的“下一步怎么打、双方往哪里移动”联合建模为序列生成问题，并把一个回合展开成带有 10 种击球关系和 2 种移动意图关系的 Player Movements（PM）动态图。DyMF 用 relational GCN 表示回合交互、用由运动模式动态生成权重的 GCN 表示球员风格，再通过两级融合预测击球类型及双方位置分布。在 4,325 个真实回合上，DyMF 对三种观测长度均取得表中最优结果；其真正有价值之处是把“击球—接球移动—回防移动”写进图结构，而不只是换了更强的序列编码器。

## 毒舌评论

这是一篇“问题定义和结构归纳偏置比模型组件更值钱”的论文：PM graph 把运动语义编码进边，贡献清楚；但所谓最高 35.3%/21.5%/24.3% 的提升主要来自挑选较弱基线作分母，相对最强基线 ShuttleNet 的优势通常只有几个百分点。再加上 best-of-10 取最优样本的宽松评估、主文未展示消融、案例研究只有一个反事实样例，“可辅助教练发现隐式战术”的应用叙事明显跑在证据前面。

## Research Question

- 研究对象：羽毛球单打回合中，两名球员的连续位置、击球者身份与击球类型。
- 小领域范围：交替行动（turn-based）运动中的联合动作—轨迹预测。
- 具体问题：给定前 \(\tau\) 拍双方位置和上一拍击球类型，联合生成后续每拍的击球类型与双方二维位置，直至回合结束。
- 为什么重要：球员移动既决定能否及时触球，也暴露回防习惯和攻防选择；仅预测击球类型或落点无法完整模拟战术。
- 论文边界：只研究单打、固定 31 名数据集内球员、最长 35 拍；不预测回合何时结束，也未验证跨球员、跨赛事或双打泛化。

## Motivation and Basic Idea

- Motivation：普通 Seq2Seq 会把回合压成线性序列，难以显式区分“对手的击球迫使我去接球”和“我击球后回到防守位”这两类语义不同的移动；已有图模型又没有适配交替击球结构，也难以表示球员在回合内不断变化的策略。
- Basic idea：把每一拍的双方位置建成两个新节点，用击球类型、`Return`、`Defend` 三类语义边连接相邻时刻；在图上分别编码回合交互与个人动态风格，再按当前局面自适应融合。
- 这个 idea 如何回应 motivation：PM graph 先把运动学相邻关系提升为战术关系；relational GCN 读取“发生了什么交互”，dynamic GCN 读取“这个球员最近如何打”，hierarchical fusion 决定“此刻更该信谁的风格、还是信回合结构”。
- 作者给出的证据：普通 Seq2Seq/Transformer 在位置指标上逊于使用 PM graph 的 GCN 变体；完整 DyMF 又优于 PM graph 上的 GCN、R-GCN 和 EvolveGCN，也优于专为羽毛球击球预测设计的 ShuttleNet。
- 我的判断：从问题到 PM graph 的推导自然且可信；从“动态战术”到 Conv1D+LSTM 生成 GCN 权重、再到两级门控融合则更像合理的工程组合。由于主文没有组件消融，无法确认每个复杂部件是否都必要。

## Background

- 背景：羽毛球研究已覆盖视频/传感器数据采集、击球影响分析、战术可视化和击球预测。
- 问题：实际战术不仅包含击球类型和落点，还包含击球者回防及对手接球的连续移动。
- Gap：此前没有直接的羽毛球 movement forecasting 任务，也没有将交替击球、接球移动和回防移动统一表示的图结构。

## Threat Model / Assumptions

- 参与者假设：双方严格交替击球，且球员身份已知；数据中的 31 名球员在训练阶段已有历史记录。
- 观测假设：每拍都能获得双方准确二维位置与离散击球标签；含缺失值的回合被整体过滤。
- 生成假设：击球类型先生成，再据此生成双方位置；未来位置服从各自的二维高斯分布。
- 不覆盖的情况：新球员冷启动、观测噪声、伤病/体能、比分和比赛阶段、回合终止、双打配合、球速/飞行轨迹及跨赛事分布漂移。

## Method

### 1. 任务形式化

一个回合记作

\[
R=\{[S_1,\ldots,S_{|R|}],P\},\qquad S_t=\{L_t,s_t\},
\]

其中 \(P=(p_a,p_b)\)，\(L_t=\{L_a^t,L_b^t\}\)，每个 \(L_{a|b}^t=(l_{a|b,x}^t,l_{a|b,y}^t)\in\mathbb{R}^2\)。输入为长度 \(\tau\) 的观测序列 \([\{L^1,s_0\},\ldots,\{L^\tau,s_{\tau-1}\}]\)，输出余下双方位置和击球类型。击球类型索引错开一拍，是因为目标位置表示一次击球后的双方位置；\(s_0\) 用零填充。

### 2. PM graph：先把战术常识写进拓扑

- 节点：每个时刻为两名球员各建一个位置节点 \((v_a^t,v_b^t)\)，因此节点随回合增长，而非始终只有两个“球员节点”。
- 关系：共 12 类，包括 10 种击球（net shot、lob、defensive shot、smash、drop、push/rush、short service、clear、drive、long service）以及 `Return` 和 `Defend`。
- 每新增一拍，加入两个节点和三条关键边：一条击球边连接当前击球者与下一位接球者；`Defend` 连接击球者击球前后的位置；`Return` 连接下一位接球者的前后位置。
- 解码时每拍分五步：先加入接球者节点与 `Return` 边；预测击球类型；再加入击球者回防节点、`Defend` 边和预测的击球边；预测双方位置；最后更新本拍涉及的四个节点表示。

这一设计是全文最关键的归纳偏置：同样是相邻时刻位移，接球与回防被赋予不同关系，从而允许关系专属消息传递。

### 3. 节点嵌入与 relational GCN

位置与球员身份被拼接为初始节点表示：

\[
e_{a|b}^t=W^e\big(\operatorname{ReLU}(W^L L_{a|b}^t)\;\|\;W^p p_{a|b}\big). \tag{1}
\]

对 PM graph 使用带自连接的 R-GCN：

\[
z_i^{(l+1)}=\sigma\left(\sum_{r\in\mathcal R}\sum_{j\in\mathcal N_r^i}a_r^iW_r^{(l)}z_j^{(l)}+W_o^{(l)}z_i^{(l)}\right), \tag{2}
\]

并用 basis decomposition \(W_r^{(l)}=\sum_{b=1}^{B}c_{rb}^{(l)}M_b^{(l)}\) 控制 12 种关系的参数量与训练稳定性。

### 4. Dynamic GCN：用最近运动模式改变球员风格编码

作者先把玩家身份再次并入节点表示，经 Conv1D 提取局部运动模式，再由 LSTM 累积长期模式：

\[
C_{a|b}^t=\operatorname{Conv1D}(E_{a|b}^t),\qquad
h_{a|b}^{(s+1)}=\operatorname{LSTM}(h_{a|b}^{(s)},c_{a|b}^{(s)},C_{a|b}^{t(s)}). \tag{5--6}
\]

隐状态序列 \(Q_{a|b}=[h^{(1)},\ldots,h^{(t)}]\) 被用作 GCN 的动态权重，得到个人风格上下文 \(d_{a|b}^i\)。直觉是：球员风格不应是整场固定 embedding，而应随刚刚出现的移动模式而改变。

### 5. Hierarchical fusion：先融合双方风格，再融合风格与回合

- Player–Player fusion：对双方动态风格序列做 parallel co-attention，得到对手对本方的全局影响门 \(f_a,f_b\)，再计算 \(D_a'=f_bD_b+D_a\)、\(D_b'=f_aD_a+D_b\)。
- Player–Rally fusion：由个人风格 \(d'^t\) 和 relational GCN 上下文 \(z^{t(l_{last})}\) 分别生成 sigmoid 门 \(\alpha,\beta\)，然后

\[
\hat e_{a|b}^t=\alpha_{a|b}d_{a|b}'^t+\beta_{a|b}z_{a|b}^{t(l_{last})}. \tag{12}
\]

注意这里不是和为 1 的 convex combination；\(\alpha\) 与 \(\beta\) 独立，模型可以同时放大或压低两路信息。

### 6. 联合预测与损失

- 击球类型：拼接当前相关的双方节点表示后做线性分类；奇偶拍交换击球者与防守者的顺序。
- 双方位置：由拼接表示分别输出两个二维高斯的五个参数 \(\{\mu_x,\mu_y,\sigma_x,\sigma_y,\rho\}\)，再从中采样下一时刻坐标。
- 总损失：

\[
\mathcal L=\mathcal L_{shot\ type}+0.5\mathcal L_{location\ a}+0.5\mathcal L_{location\ b}, \tag{15}
\]

其中击球类型用交叉熵，位置用二维高斯负对数似然。

### 关键设计取舍

- 为什么不是纯序列：它无法显式区分不同关系和移动目的。
- 为什么不是固定两节点动态图：dNRI 的结果提示只保留两个球员节点会丢失逐拍位置历史；PM graph 为每个时刻创建新节点。
- 为什么输出分布而不是单点：未来移动具有多模态不确定性。不过单个二维高斯本身仍是单峰，真正的多样性主要依赖递归采样。

## Evaluation

- 数据：过滤所有含缺失值回合后，75 场比赛、180 局、4,325 个回合、43,191 次击球、31 名高水平球员；回合最长截断为 35。
- 切分：每场比赛按回合顺序，前 80% 训练、后 20% 测试，使模型可以使用同一球员更早的历史。该设置不是跨球员泛化测试。
- 基线：Seq2Seq、Transformer trajectory forecasting（TF）、ShuttleNet；dNRI、GCN\(_{PM}\)、R-GCN\(_{PM}\)、E-GCN\(_{PM}\)。后三者由作者移植到 PM graph 和 encoder–decoder 框架。
- 观测长度：\(\tau\in\{2,4,8\}\)。
- 指标：双方位置的 MSE、MAE，以及击球类型的 cross-entropy（CE），均为越低越好。
- 随机性处理：每个回合生成 10 条序列，选择离真值最近的一条用于评价；表中为 5 个随机种子的平均值。
- 训练：100 epochs，Adam，batch size 32，学习率 0.001；5-fold cross-validation 调参；GTX 2070 8GB。

### 主要结果

| 观测长度 | DyMF MSE | 最强对手 | DyMF MAE | 最强对手 | DyMF CE | 最强对手 |
|---|---:|---:|---:|---:|---:|---:|
| \(\tau=2\) | **1.1006** | ShuttleNet 1.1345 | **1.5795** | ShuttleNet 1.6179 | **1.9543** | ShuttleNet 1.9599 |
| \(\tau=4\) | **1.1061** | ShuttleNet 1.1539 | **1.5888** | R-GCN\(_{PM}\) 1.6339 | **1.9487** | ShuttleNet 1.9575 |
| \(\tau=8\) | **1.0827** | ShuttleNet 1.1399 | **1.5739** | ShuttleNet 1.6360 | **1.9570** | ShuttleNet 1.9654 |

- DyMF 在九个“观测长度 × 指标”组合上均排名第一，平均排名 1.0。
- 相对各列最强基线，MSE 约改善 3.0%–5.0%，MAE 约改善 2.4%–3.8%，CE 约改善 0.29%–0.43%。这比论文强调的“最高 35.3% / 21.5% / 24.3%”更能反映与强基线的实际差距。
- E-GCN\(_{PM}\) 随观测长度增长发生严重数值退化（\(\tau=8\) 的 MSE 为 1314.0），作者推测短图导致状态不稳定，但没有进一步诊断。
- 案例研究人为改变 Player A 的回防位置，DyMF 随之把 Player B 的预测从网前球改为高远球，展示了反事实分析的可能性；这不是因果验证，也未由教练或真实比赛结果确认。

## Key Artifacts

- Figure 1：说明一次击球后双方位置和下一拍类型存在多种可能，支撑联合概率预测而非单一分类。
- Figure 2：DyMF 总体架构；最重要的信息流是 PM graph → relational/dynamic GCN → player-player/player-rally fusion → 击球与位置双头预测。
- Figure 3：编码、解码阶段的 PM graph 构造，是复现时最关键的图；明确了每拍节点/边的添加顺序与自回归依赖。
- Table 1：唯一的主结果表，支撑 DyMF 在全部设置排名第一，但也显示对最强基线的绝对优势有限。
- Figure 4：修改回防位置后的预测变化，支撑模型可用于“what-if”战术探索；证据等级仅为单案例演示。
- Equations (2)–(3)：12 类关系上的 R-GCN 与 basis decomposition，承担结构交互建模。
- Equations (5)–(7)：Conv1D + LSTM 生成动态图卷积权重，承担回合内风格变化建模。
- Equations (8)–(12)：两级融合，承担对手风格影响与回合/风格权衡。
- Equations (14)–(17)：二维高斯位置头与联合损失，定义模型真正优化的目标。

## Findings

- 发现 1：在这个任务中，显式编码 `Return` 与 `Defend` 比只把位置当作普通轨迹点更有效；GCN\(_{PM}\)/R-GCN\(_{PM}\) 整体优于朴素序列基线。
- 发现 2：为每拍创建位置节点比只保留两个球员实体节点更合适；dNRI 在所有设置中表现很弱。
- 发现 3：DyMF 的优势随更长观测窗口没有消失，说明动态风格与层次融合至少在表中具有一致性，而非只适用于短上下文。
- 发现 4：击球 CE 的增益远小于位置误差增益，暗示主要收益更可能来自移动结构建模，而不是击球分类能力的大幅提升。

## Strengths

- 问题定义有实际意义：把击球与双方移动统一预测，比单纯的 stroke forecasting 更贴近完整战术状态。
- PM graph 简洁且有领域语义，节点、边和解码顺序都可解释，也较容易迁移到网球等交替击球运动。
- 基线覆盖序列、轨迹 Transformer、羽毛球专用模型和多类图模型；三种观测长度下结果方向一致。
- 输出位置分布而不是单点，承认体育行为的随机性，并给出可运行代码链接和完整训练环境。

## Limitations

- 主文称消融在 Appendix C.1，但所提供的 8 页 PDF 不含附录；因此无法核验 dynamic GCN、两级 fusion、各关系类型分别贡献多少。
- best-of-10 评价会奖励采样覆盖而非概率校准，且没有报告 ADE/FDE、NLL、top-k accuracy、置信区间或显著性检验；不同模型的分布锐度可能被掩盖。
- 训练/测试都来自同一批比赛与球员，球员 embedding 可能记住个体；论文没有 leave-one-player-out 或跨赛事评估。
- 删除所有含缺失值的回合可能造成选择偏差；未报告删除比例与坐标标注噪声。
- 单个二维高斯难以表示明显多峰的下一位置；递归采样会累积击球类型和位置误差。
- 论文没有显式建模回合终止，任务定义默认已知未来长度或在固定真实回合范围内评价；真实部署时“何时结束”本身也需预测。
- Figure 4 只有一个模型反事实案例，没有真实反事实标签、专家评审或训练收益实验，不能证明建议具有因果效应或实战价值。
- EvolveGCN 基线灾难性失稳拉高了论文的“最大提升”，但作者没有排查实现、超参数或数值稳定性，不能把该差距全部归因于方法优越性。
- 仅单打、31 名高水平球员和一个数据集，外部有效性有限；双打中的队友协同与更复杂遮挡并非小改动即可解决。

## My Takeaways

- 对视觉/运动分析研究的启发：如果轨迹点之间存在明确的行为目的，应把目的编码成关系类型，而不是让通用时序模型从坐标中自己猜。
- 可复用的方法：将每时刻实体状态实例化为新节点；用“跨实体动作边 + 同实体状态转换边”构成时序多关系图。
- 复现优先级：先实现 PM graph + R-GCN 与二维高斯头，确认表 1；再逐步加入 dynamic GCN、player-player fusion 和 player-rally fusion，避免一次复刻全部复杂组件。
- 更强的后续方案：用 mixture density network / diffusion 表示多峰位置分布；加入 rally termination head；做球员冷启动、跨赛事和按比赛留出的评估；报告 expected/minADE@K、NLL 与 calibration。
- 可做的研究问题：将 PM graph 与视频端到端位置估计相连，系统性分析观测噪声如何传递到战术预测，而不是只在干净人工标注上评估。

## Related Papers

- 前置阅读：Wang et al. (2022b), *ShuttleNet: Position-Aware Fusion of Rally Progress and Player Styles for Stroke Forecasting in Badminton*；它是最接近的任务与最强基线。
- 图建模基础：Schlichtkrull et al. (2018), *Modeling Relational Data with Graph Convolutional Networks*；Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*。
- 动态图对比：Pareja et al. (2020), *EvolveGCN*；Graber & Schwing (2020), *Dynamic Neural Relational Inference*。
- 轨迹预测对比：Giuliari et al. (2020), *Transformer Networks for Trajectory Forecasting*。
- 关键差异：ShuttleNet 固定融合回合进程与球员风格，DyMF 则显式建图、区分接球/回防语义，并让球员风格和融合权重随回合变化。

## Open Questions

- 若去掉球员身份 embedding，仅用近期运动，DyMF 还能否泛化到未见过的新球员？
- `Return`/`Defend` 两类边各自贡献多少？把击球类型边简化为若干战术簇会不会更稳健？
- best-of-10 下第一的模型，在 NLL、概率校准和单次采样质量上是否仍然第一？
- 论文评价是否使用真实未来回合长度？如果是，加入终止预测后性能会下降多少？
- Figure 4 的反事实变化是真正学到位置约束，还是球员/击球共现统计造成的相关性？
- 单峰二维高斯换成混合高斯、normalizing flow 或 diffusion 后，是否能更真实地覆盖多种落位选择？
- PM graph 能否直接迁移到网球、乒乓球，还是需要重新定义回防与击球关系？

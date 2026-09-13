# 【arXiv‘2023】Siamese Masked Autoencoders

## Metadata

- Title: Siamese Masked Autoencoders
- Authors: Agrim Gupta, Jiajun Wu, Jia Deng, Li Fei-Fei
- Venue / Year: arXiv preprint, 2023（论文版本 v1，2023-05-23）
- Note Name: 【arXiv‘2023】siamese-masked-autoencoders
- Paper: [arXiv:2305.14344](https://arxiv.org/abs/2305.14344)；本地文件：[2305.14344v1.pdf](D:\learning\badminton\参考文献\2305.14344v1.pdf)
- Code: TBD；论文仅说明基于 MAE 开源实现
- Dataset / Artifact: Kinetics-400；下游评估使用 DAVIS-2017、JHMDB、VIP
- Scope / Subfield: 视频自监督表示学习、时空稠密对应（dense correspondence）
- Tags: `Masked Autoencoder`, `Video Representation Learning`, `Temporal Correspondence`, `Siamese Network`, `Vision Transformer`, `Self-supervised Learning`
- Status: DONE

## TL;DR

论文提出 SiamMAE，用“过去帧完整输入、未来帧随机遮挡 95%”的非对称掩码，把普通 MAE 的重建任务改造成时空对应学习任务。两帧由共享权重的 Siamese ViT 独立编码，再由 cross-attention + self-attention 解码器从过去帧推断未来帧的缺失 patch。该方法不需要对比学习的复杂增强、负样本或防坍塌组件，在 DAVIS 视频目标分割、JHMDB 姿态关键点传播和 VIP 语义部件传播上取得很强的 zero-shot label propagation 结果。

## 毒舌评论

这篇工作的真正贡献不是“又做了一个 MAE”，而是抓住了视频与图像最不该被同等处理的地方：时间方向有因果结构，空间方向没有。95% 的未来帧掩码很有效，但论文把“对应关系来自非对称预测”与“提升也可能来自更小 patch、更长训练、特定 kNN 推理设置”混在了一起；尤其最佳结果依赖 ViT-S/8 和 2000 epoch，默认 ViT-S/16 反而明显落后于 VFS。换句话说，idea 很漂亮，证据也不错，但“简单且全面胜出”的叙事比表格本身更大胆。

## Research Question

- 研究对象：从无标签视频中学习可用于跨帧匹配的视觉特征。
- 小领域范围：视频时空稠密对应、对象级跟踪与标签传播。
- 具体问题：如何在不使用人工光流/跟踪标注、不依赖精心设计的数据增强和对比损失的情况下，学习能处理运动、遮挡、视角变化和外观变化的 correspondence features？
- 为什么重要：稠密对应是视频目标分割、姿态传播、语义部件传播、光流、深度和三维重建的共同基础。
- 论文边界：只研究两个帧之间的预测；不系统研究多未来帧、模型/数据规模扩展、不同视频域，也不直接解决完整的在线跟踪系统。

## Motivation and Basic Idea

- Motivation：
  - 传统预测式自监督方法可以从参考帧预测未来帧，但性能曾落后于对比学习。
  - 图像 MAE 的目标偏向像素重建，特征在 zero-shot correspondence 上不一定好。
  - VideoMAE、MAE-ST 等方法对所有帧采用近似对称掩码，可能让模型主要利用单帧的空间冗余，而不是学习“物体从哪里移动到哪里”。
  - 视频的时间维度具有特殊结构，不能简单视作又一个空间维度。
- Basic idea：给定过去帧 `f1` 和未来帧 `f2`，完整保留 `f1`，随机遮挡 `f2` 的 95% patch；让模型只能大量依赖 `f1` 的特征来预测 `f2`，从而学习跨帧对应。
- 这个 idea 如何回应 motivation：
  - 低掩码比例时，模型可用未来帧残留 patch 和图像局部冗余完成重建，不必建立对应。
  - 对称掩码时，两帧都信息不足，模型的训练目标更像普通时空重建。
  - 非对称掩码把任务偏置到“从过去向未来传播信息”，直接把 correspondence 放进预训练压力里。
- 作者给出的证据：
  - Table 2：非对称 95% 掩码优于对称掩码；Siamese encoder + cross-self decoder 最好。
  - Table 3：随机扩大帧间隔、减少颜色增强有利于结果。
  - Fig. 5：CLS token 的 self-attention 在 ImageNet 图像上呈现出物体边界。
- 我的判断：这是一个很强的任务设计（objective design）贡献。它不是显式监督匹配，而是通过信息不对称构造出“必须匹配才容易预测”的瓶颈。不过论文没有完全证明模型一定在学习真实物体对应，也可能部分学习了 Kinetics 中常见动作、构图和局部外观统计。

## Background

- 背景：光流和像素级运动估计通常需要人工或合成的像素级监督；对象级 correspondence 更适合通过视频自监督学习。
- 问题：对比式方法效果强，但依赖增强设计，并通常需要额外机制避免表示坍塌。
- Gap：已有 MAE 视频扩展主要对所有帧对称掩码，未充分利用过去到未来的方向性；预测式学习在 correspondence 上的潜力没有被充分发挥。

## Threat Model / Assumptions (Optional)

- 攻击者或参与者能力：N/A。本文不是安全论文。
- 链上 / 链下假设：N/A。
- 关键建模假设：
  - 视频在相邻或中等时间间隔内具有足够的时空连续性。
  - 过去帧包含未来帧中多数可预测对象的信息。
  - 随机裁剪后的两帧仍处在可建立对应的视觉关系中。
- 不覆盖的情况：
  - 物体突然出现或消失、剧烈镜头切换、长时间遮挡、非刚性快速运动。
  - 未来具有多个同样合理结果时，像素级 L2 重建的平均化问题。
  - 与训练分布差异很大的 egocentric、工业或体育视频。

## Method

- 核心思路：
  1. 从一个视频 clip 中随机选两帧 `f1`、`f2`，`f2` 是未来帧。
  2. 将每帧切成不重叠的 `N x N` patch，加入空间位置编码和 `[CLS]` token；不使用 temporal position embedding。
  3. `f1` 保留 0% 掩码，`f2` 掩码 95%。
  4. 两帧分别输入共享权重的 Siamese ViT encoder。
  5. decoder 中，`f2` token 通过 cross-attention 读取 `f1` token，再通过 self-attention 让 `f2` 的预测保持空间一致。
  6. 只对 `f2` 被遮挡 patch 预测归一化像素值，并计算 L2 reconstruction loss。
  7. 预训练后丢弃 decoder，直接使用 encoder 特征进行跨帧 kNN label propagation。
- 核心思路是怎么想到的：
  - MAE 已经证明高比例掩码能迫使模型提取全局结构。
  - 视频额外提供了天然的数据变化：姿态、光照、视角、遮挡和运动。
  - 将“未来帧高掩码”与“过去帧完整”组合，可以把普通重建转化为带方向的 patch 传播。
  - Siamese encoder 保证两帧由同一特征空间独立处理，避免训练时依赖跨帧联合编码而测试时无法复现。
- 从 motivation 到 method 的逻辑链：
  - 目标是 correspondence，而不是单帧像素重建。
  - 需要让单帧空间捷径失效，因此提高未来帧遮挡率。
  - 需要让模型学到跨帧共享表征，因此使用权重共享、独立编码。
  - 需要显式建立 source-target 关系，因此 decoder 使用 cross-attention。
  - 需要让目标帧预测不至于 patch-wise 独立破碎，因此 cross-attention 后加入 self-attention。
- 关键设计取舍：
  - `joint encoder`：两帧 token 拼接后联合编码；实现直接，但与测试时逐帧独立提特征不一致。
  - `siamese encoder`：共享权重、独立处理；与下游评估一致，并形成信息瓶颈。
  - `joint decoder`：全 token 联合 self-attention；显存开销大。
  - `cross decoder`：只有 f2 读 f1；预测空间上不连贯。
  - `cross-self decoder`：先跨帧匹配，再在目标帧内部整合；实验中最好。
- 为什么不是更直接 / 更简单的方案：
  - 只用对称掩码会给模型留下利用单帧空间冗余的路径。
  - 只用 cross-attention 会让每个目标 patch 独立预测，缺少目标帧内部的空间协调。
  - 只用 joint encoder 会让预训练接口与下游单帧特征提取不匹配。
- 系统流程或算法步骤：
  - 输入：Kinetics-400 中一段视频，随机帧间隔 `g in [4, 48]`，采样率 30 fps，即约 0.13 到 1.6 秒。
  - 预训练分辨率：`224 x 224`；主干：ViT-S/16，21M 参数；另实验 ViT-S/8。
  - 编码：`z1 = E_theta(f1)`，`z2 = E_theta(mask(f2))`，其中 `E_theta` 两路共享参数。
  - 解码：`h2 = D_cross-self(z2, z1)`。
  - 预测：`y_hat2 = P(h2)`，只在 mask 集合 `M2` 上计算损失。
  - 目标函数可写为：
    `L = (1 / |M2|) * sum_{i in M2} || y2_i - y_hat2_i ||_2^2`
  - 下游：用 encoder patch features 做 kNN 匹配，将初始帧标签传播到未来帧；推理时维护最近 20 帧的 context queue，并限制 source patch 的空间邻域。
- 关键定义 / 公式 / 不变量：
  - `f1`：过去/参考帧；`f2`：未来/目标帧。
  - asymmetric masking：`r(f1)=0%`，`r(f2)=95%`。
  - cross-attention 中，`f2` token 是 query，`f1` token 提供 key/value；其注意力矩阵在功能上类似 correspondence 方法中的 affinity matrix。
  - 论文正文没有给出更完整的 attention 展开公式；上面的损失形式是根据文字描述对 MAE 目标的规范化重写。
- 实现细节：
  - 优化器 AdamW，`beta1=0.9`、`beta2=0.95`，weight decay 0.05，学习率 `1.5e-4`，cosine decay，40 epoch warmup。
  - batch size 2048；主结果训练 2000 epoch，消融训练 400 epoch；repeated sampling factor 为 2。
  - 数据增强为 random resized crop（scale `[0.5, 1]`）和 horizontal flip；颜色抖动反而降低性能。
  - ViT-S/16 使用 4 张 Titan RTX，ViT-S/8 使用 8 张 Titan RTX。

## Evaluation

- 实验思路：
  - 不对下游模型进行微调，使用预训练 encoder 的 patch features 做 kNN label propagation，直接测量 correspondence 能力。
  - 三个任务覆盖不同精细程度：DAVIS 多目标分割、JHMDB 15 个姿态关键点传播、VIP 20 类人体语义部件传播。
  - 同时比较 ImageNet/视频自监督、对比学习和视频 correspondence 方法，并对 encoder、decoder、掩码、增强、帧间隔、训练时长和 patch size 做消融。
- 评估指标：
  - DAVIS：`J&F mean`，其中 `J` 为区域重叠，`F` 为边界准确性。
  - VIP：`mIoU`。
  - JHMDB：`PCK@0.1`、`PCK@0.2`。
- 主要结果：
  - 主表中最好的 SiamMAE ViT-S/8：DAVIS `71.4 J&F`、VIP `45.9 mIoU`、JHMDB `61.9/83.8 PCK`。
  - 相比 DINO ViT-S/8：DAVIS `69.9 -> 71.4`，VIP `39.5 -> 45.9`，JHMDB `56.5/80.3 -> 61.9/83.8`。
  - SiamMAE ViT-S/16：DAVIS `62.0`、VIP `37.3`、JHMDB `47.0/76.1`，仅略高于 DINO ViT-S/16 的 DAVIS `61.8`，且不如表中的 VFS `68.9/43.2/60.9/80.7`。
  - 与 VideoMAE ViT-S/16 相比，DAVIS 从 `39.3` 提升到 `62.0`，支持“对称 tube masking 不利于 correspondence”的解释，但这仍是跨方法比较，不能单独视为因果证明。
  - 消融：默认 Siamese + cross-self + asymmetric 95% 在 DAVIS 达到 `58.1 J&F`；相同设置下 joint encoder / joint decoder 为 `49.7`，siamese encoder / cross decoder 为 `52.2`。
  - 非对称掩码比例从 50%、75%、90% 到 95% 时，DAVIS `J&F` 为 `49.0、55.3、58.4、58.1`；95% 并非严格最高，但整体显示高比例掩码更有利。
  - 随机帧间隔 `[4,48]` 的 `58.1` 高于固定 4 帧的 `55.1`、固定 8 帧的 `56.4`、固定 16 帧的 `58.0` 和固定 32 帧的 `57.7`。
  - 训练越久、patch 越小，三个下游任务都持续改善；这说明性能仍有明显 compute/data scaling 成分。
  - 颜色增强加入后 DAVIS 从 `58.1` 降到 `55.8`；仅空间增强的配置最好，说明视频中的自然时序变化已经提供了主要不变性来源。

## Key Artifacts

- 关键图：
  - Fig. 1：SiamMAE 结构图，支撑“过去帧完整输入、未来帧高比例掩码、两路独立编码、cross-attention 解码”的核心流程。
  - Fig. 2：Kinetics-400 重建可视化，支撑模型能根据过去帧恢复未来帧的部分结构；它是定性证据，不等同于 correspondence 的定量证明。
  - Fig. 3：DAVIS、JHMDB、VIP 的标签传播定性结果，支撑特征能传播对象、关键点和语义部件。
  - Fig. 4：训练 epoch 与 patch size 曲线，支撑更长训练和更小 patch 带来稳定收益，也暴露了计算规模对结论的重要性。
  - Fig. 5：CLS self-attention 边界可视化，支撑“物体边界涌现”的观察；但只有单 head、定性样例，不能说明所有场景都具备边界分割能力。
- 关键表：
  - Table 1：跨方法主结果，最重要的结论是 ViT-S/8 SiamMAE 在三类 label propagation 任务上优于表中对比方法。
  - Table 2a：FrameMAE 的对称掩码结果，说明简单把 MAE 扩展到两帧并不能得到好的 correspondence。
  - Table 2b：encoder-decoder 组合，说明 Siamese encoder 与 cross-self decoder 的组合最有效。
  - Table 2c/2d：对称/非对称掩码和比例，说明任务难度与对应关系诱导之间存在合适的区间。
  - Table 3：手工增强与自然帧采样，说明大时间间隔比颜色抖动更有价值。
  - Table 4：训练与评估超参数，支撑复现，但没有给出完整代码和随机种子。
- 关键公式 / 定义 / 算法：
  - `r(f1)=0%`、`r(f2)=95%`：非对称掩码是论文最核心的 inductive bias。
  - `f2 query -> f1 key/value`：cross-attention 提供源帧到目标帧的匹配通道。
  - mask-only L2 reconstruction：把学习压力集中到未来帧未知 patch。
- 这些证据分别支撑哪些结论：
  - Table 2 支撑“组件设计确实有贡献”。
  - Table 1 支撑“表内 benchmark 上的迁移有效性”。
  - Fig. 4 支撑“训练时长和 patch size 是重要变量”。
  - Fig. 5 支撑“边界注意力是一个有趣的涌现现象”，但不是严格机制验证。

## Findings

- 发现 1：预训练任务的方向性比“视频化 MAE”本身更重要。视频不是把多帧堆起来就够了，必须明确规定哪些信息可见、哪些信息必须从历史推断。
- 发现 2：Siamese encoder 和 cross-self decoder 形成了与下游 label propagation 相容的接口：encoder 独立产出可比较特征，decoder 的 cross-attention 又显式训练了 source-target 关系。
- 发现 3：自然视频变化可以替代一部分手工增强。扩大帧间隔带来的姿态、光照、视角和遮挡变化，比颜色抖动更贴近 correspondence 的实际不变性。
- 发现 4：细粒度对应高度依赖 patch size。SiamMAE ViT-S/8 相比 ViT-S/16 的提升不是小修小补，而是结果能否超过强 baseline 的关键因素。
- 发现 5：像素重建目标并不必然只学低级纹理。合适的时序瓶颈可能让 CLS attention 自发聚焦物体边界，但论文没有提供边界检测的系统定量评估。

## Strengths

- 论文抓住了视频建模中“时间维度具有方向性”的核心事实，并把它落实为非常简单、容易复现的掩码策略。
- 消融覆盖了最重要的结构变量，尤其清楚地展示了 joint/siamese encoder、cross/cross-self decoder 和 mask ratio 的交互。
- 评估采用 zero-shot kNN label propagation，直接测试特征是否具备对应能力，避免下游微调掩盖预训练表示的差异。
- 三个下游任务从对象区域、人体关键点到语义部件，覆盖了不同空间粒度；ViT-S/8 的结果在表内比较中有竞争力。
- 方法不需要额外的负样本、复杂防坍塌机制或 tracking-based handcrafted pretext task，工程实现相对清楚。

## Limitations

- 主结果依赖 ViT-S/8、2000 epoch 和较大的 batch size；“简单、低成本”的印象需要结合训练资源重新评估。
- 默认 ViT-S/16 并未全面优于 VFS；因此论文“全面超过 state-of-the-art”的结论应限定为最佳 patch-size 配置和表内 benchmark。
- 预训练只使用 Kinetics-400，且下游任务也偏向视频中的人物/对象传播，跨领域泛化没有验证。
- 只预测两帧，无法说明该方法对长时序、累积漂移、循环一致性或多未来分支是否有效。
- 未来帧可能有多种合理状态，单一像素级 L2 目标可能鼓励平均化预测；论文没有比较感知损失、特征损失或离散 token 目标。
- 下游推理使用了 context queue、空间邻域和 top-k 等超参数；附录说明对部分 baseline 做了 comprehensive grid search，比较可能对评估设置敏感。
- 没有公开完整代码、训练日志、随机种子或逐项计算成本；复现仍可能依赖 MAE 实现细节。
- Fig. 5 的 object boundary 结论来自少量定性可视化，且只展示单个 attention head，机制解释仍偏推测。

## My Takeaways

- 对视频自监督而言，最值得复用的不是某个 Transformer block，而是通过可见性设计制造正确的信息瓶颈。
- 对羽毛球视频分析尤其有启发：可以把击球前完整帧作为 reference，把击球后或运动中间帧高比例遮挡，迫使模型学习球拍、手臂、身体和羽毛球的运动对应，而不是只记住外观。
- 但羽毛球动作存在高速小目标、强遮挡、镜头切换和非刚性形变，直接使用 95% 掩码可能过于困难；更合理的后续实验是按目标尺度、帧间隔和动作阶段自适应掩码。
- 可复用的方法：
  - 先做“方向性预训练任务”设计，再选择 backbone。
  - 用 patch-level feature matching 验证 correspondence，而不只看分类准确率。
  - 把自然时序变化视为增强来源，分别控制空间增强、颜色增强和帧间隔。
  - 用 decoder 的 cross-attention 或 encoder 特征相似度可视化检查模型究竟在匹配什么。
- 可能的后续问题：
  - 多个未来帧是否能学习更稳定的长时运动表示？
  - 能否把像素 L2 换成目标检测/关键点/特征空间预测？
  - 对高速羽毛球、球拍和手腕等小目标，patch size 是否应进一步缩小或采用多尺度 patch？
  - 是否可以加入双向或 cycle consistency，同时避免重新引入复杂的对比学习组件？

## Related Papers

- 前置阅读：
  - He et al., *Masked Autoencoders Are Scalable Vision Learners*：图像 MAE 的随机高比例掩码基础。
  - Tong et al., *VideoMAE*：对称视频掩码的代表性方法。
  - Feichtenhofer et al., *Masked Autoencoders as Spatiotemporal Learners*：视频时空 MAE 的另一条路线。
- 后续阅读：
  - Wang et al., *Tracking Emerges by Colorizing Videos*：预测式 correspondence 学习的早期代表。
  - Li et al., *Joint-task Self-supervised Learning for Temporal Correspondence*：视频标签传播与自监督 correspondence。
  - Xu and Wang, *Rethinking Self-supervised Correspondence Learning*：视频帧级相似性视角。
- 可对比论文：
  - Caron et al., *Emerging Properties in Self-supervised Vision Transformers*（DINO）：对比式 ViT 表示与涌现分割现象。
  - Jabri et al., *Space-time Correspondence as a Contrastive Random Walk*：显式时空 correspondence 学习。
- 最接近的相关工作：VideoMAE、MAE-ST、UVC、VFS。
- 关键差异：SiamMAE 的主要 delta 是非对称 future masking + Siamese independent encoding + cross-self decoding，而不是引入新的 backbone 或复杂损失。

## Open Questions

- 在羽毛球等高速体育视频中，帧间隔应以固定帧数、真实时间还是动作阶段来采样？
- 对小目标而言，ViT-S/8 的 patch 粒度是否仍不够；多尺度或局部高分辨率分支能否改善球和球拍对应？
- 训练时的像素重建是否真的比 feature reconstruction 更能帮助关键点传播？
- 如何定量区分“物体真实对应”与“动作类别/背景/构图捷径”？
- 在遮挡和镜头切换下，加入不确定性建模或多假设未来预测是否优于单一 L2 预测？

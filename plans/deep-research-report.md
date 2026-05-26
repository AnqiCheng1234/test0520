# 面向冻结深度基础模型的 RAW-like 单目深度残差校正文献调研

## Executive summary

你的方向**值得继续做**，但应当明确把论文定位在“**synthetic RAW-like representation 对冻结 RGB depth foundation model 的补充式校正**”，而不是“真实 sensor RAW 提供了 sRGB 之外的额外信息”。在你当前的 VKITTI + InvISP 设定下，最安全、也最有机会过审的核心问题不是 “RAW 是否替代 RGB 做深度”，而是 “**逆 ISP 得到的 RAW-like 表征，能否在不破坏 DAV2 先验的前提下，帮助一个冻结的 RGB 深度基础模型修正其失败区域**”。这一定义与 RAW-Adapter、RAM、AODRaw 一类“RAW 适配/替换输入”的工作有明显差异，也更贴合你的已有失败现象。citeturn24view0turn24view1turn29view0turn30view0turn31view0

从本次检索覆盖到的主流论文来看，我**没有找到**一篇已经明确提出并系统验证“**frozen RGB depth foundation model + RAW or synthetic RAW residual correction branch**”这一组合设定的工作。最接近的相关证据有三类：其一是 RAW object detection / segmentation 中的适配或任务驱动 ISP，如 RAW-Adapter、RAM、AODRaw；其二是深度领域里对 foundation model 做轻量传感器适配或 cross-modal prompting / distillation，例如 Depth Prompting、PromptDA、PPFT、EventDAM；其三是针对夜间/恶劣条件的 monocular depth 训练中使用 inverse-ISP / noise modeling 的工作，但这类工作通常不是把 RAW 当作测试时辅助模态。这个空缺意味着你的问题 formulation 有实际 novelty 空间。citeturn24view0turn25view0turn29view0turn30view0turn31view0turn35view0turn12search5turn33view3turn33view0turn32view0

你目前的失败结果，其实与已有文献的机制性结论是高度一致的。RAW-Adapter 和 AODRaw 都直接指出，**sRGB 预训练模型与 RAW 输入之间存在显著 domain gap**，而把 sRGB-pretrained 模型直接迁到 RAW 域往往会受限；Depth Prompting 进一步指出，**把不同模态或不同传感器偏置共同揉成 joint representation 容易对泛化不利**；PPFT 和 EventDAM 也都强调，将另一种感知模态直接塞进预训练基础模型并不容易，必须用结构化的 prompt / distillation / feature guidance 去做“受控注入”。你的 “RAW-to-3-channel 再喂 DAV2 会退化、边界和天空变糊、loss 虽降但结果不稳” 与这些观察是同方向的。citeturn24view0turn24view1turn31view0turn35view0turn34view3turn33view0

因此，我最推荐的方法路线不是让 RAW branch 自己预测完整深度，而是做成“**error-aware / uncertainty-guided gated residual correction**”：RGB 仍走 frozen DAV2 主干，RAW-like branch 只预测**哪里需要改、改多少、用什么局部仿射或局部残差去改**。这条路线最符合你的实验动机，也最容易把 synthetic RAW 的 claim 风险降到最低。类似“辅助模态只在 RGB 失败处修正”的思想，在 depth prompting、polarization 引导深度增强、radar-camera confidence fusion、event-to-depth distillation 中都有成功先例，只是它们的辅助模态不是 RAW-like。citeturn35view0turn33view3turn33view4turn33view0turn33view1

关于 synthetic RAW，结论需要说得非常严格：**你不能用 VKITTI + inverse-ISP RAW 来证明“RAW 含有 sRGB 之外的真实 sensor information”**。因为从 sRGB 反推的 RAW-like 表征不可能恢复 ISP 已经真实丢失的传感器信息。InvISP、ParamISP、Unprocessing、CycleISP 这些工作证明的是“可以从 sRGB 重建或合成一个更接近 RAW 域的表示，并在某些任务上很有用”，但不是“从 sRGB 重新获得原始传感器测量本体”。因此，你的 safest claim 应该改成：**inverse-ISP-generated RAW-like representation is an alternative sensor-inspired representation that may expose complementary cues useful for depth refinement**。citeturn37search0turn37search4turn37search1turn37search13turn23search0turn23search1turn37search2turn37search10

就会议论文潜力而言，我认为这个题目在**problem formulation、负结果分析、稳妥 claim、系统 ablation**上有发表价值，但它更像一篇**结构设计扎实、问题设定新、实验严谨**的论文，而不是“纯靠模块创新取胜”的论文。你最强的卖点会是：第一，指出 RAW-as-main-input 会破坏 foundation depth prior；第二，提出保留 RGB 主路径、让 RAW-like 只做 failure-region correction 的 formulation；第三，系统证明在 synthetic RAW 设定下，补充式表征比替代式表征更稳。citeturn24view0turn29view0turn31view0turn35view0turn12search0

## Taxonomy and most relevant papers

### Taxonomy

从你的问题出发，相关文献可以被整理成七个桶。第一桶是 **RAW for vision**，关注 RAW 相对 sRGB 的信息差异、任务驱动 ISP、RAW detection / segmentation / classification；第二桶是 **inverse ISP / synthetic RAW generation**，关注如何从 sRGB 反推 RAW 或构造 RAW-like 数据；第三桶是 **RAW-to-pretrained-model adaptation**，关注如何把 sRGB 预训练模型迁到 RAW 域；第四桶是 **RAW or sensor-level data for depth**，这一桶在 monocular depth 上非常稀疏，更多是夜间深度、event depth、thermal depth 或 depth enhancement；第五桶是 **depth foundation model adaptation**，关注轻量 adapter、prompt、distillation、relative-to-metric adaptation；第六桶是 **residual / refinement depth methods**，聚焦 base prediction + refinement / boundary sharpening / uncertainty-guided correction；第七桶是 **auxiliary modality fusion for depth**，研究一类辅助模态如何只在 RGB 不可靠时提供补充。citeturn24view0turn29view0turn37search0turn35view0turn33view0turn33view3turn33view4

这七个桶里，对你最重要的不是“有没有人做过 RAW depth”，而是三条跨桶逻辑。第一，**RAW 表征在 detection / segmentation 等任务里确实能有用**，但成功做法大多围绕 ISP-learning、预训练适配、或 RAW-pretraining，而不是“原地拿 RGB foundation model 直接吃 RAW-like 3-channel”；第二，**foundation depth model 的强先验非常宝贵**，已有深度工作更倾向用 prompt、few-parameter tuning、distillation 或小修正模块，而不是大幅改输入分布；第三，**辅助模态最稳妥的角色往往是修正主模态的失败区域**，而不是整图替代主模态。你的方向正好落在这三条逻辑交叉处。citeturn24view1turn31view0turn35view0turn12search5turn33view3turn33view4turn33view0

### Most relevant papers table

| Paper | Year | Venue | Task | Input modality | Real RAW or synthetic RAW | Uses sRGB pretrained model | Frozen or fine-tuned | Fusion level | Main idea | Relevance to your idea | Limitation for your setting |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| RAW-Adapter citeturn24view0turn24view1 | 2024 | ECCV | Detection + segmentation | Camera RAW + learnable ISP adapters | PASCAL RAW / LOD 为真实 RAW；ADE20K RAW 为 InvISP 合成 | 是 | 主要是适配预训练 backbone，而非完全冻结 | Input + feature | 用 input-level ISP adapter 与 model-level adapter 适配 sRGB 预训练模型 | 与你最接近的“RAW 适配预训练视觉模型”基线 | 仍然在“让 RAW 进入主模型”的范式内，不是“保留 RGB 主路径 + RAW 补充校正” |
| Beyond RGB / RAM citeturn29view0turn30view0 | 2025 | ICCV | RAW object detection | RAW Bayer / RGGB-stacked | 真实 RAW 数据集为主 | 部分实验使用冻结预训练 detector | 端到端或冻结 detector 均有 | Input-level pre-processing | 用并行 ISP 函数代替传统 ISP，为 detector 学习任务最优前端 | 证明“lightweight RAW front-end + frozen pretrained detector”可行 | 任务是 detection，不是 dense depth；优化目标更宽容，不直接支持 depth 边界/尺度一致性 |
| Towards RAW Object Detection in Diverse Conditions / AODRaw citeturn31view0 | 2025 | CVPR | RAW object detection benchmark | Real RAW + sRGB | 真实 RAW | 是，但指出 sRGB pretrain 受限 | 预训练 + fine-tune；提出 RAW pretraining | Training / representation | 系统说明 sRGB-pretrain 到 RAW 的 domain gap，并用 RAW pretraining + distillation 缓解 | 对你解释 DAV2 输入分布敏感性非常关键 | 仍然不是 depth，也不讨论 residual correction |
| Dirty Pixels citeturn26search0turn26search2turn26search3 | 2021 | TOG / SIGGRAPH | Classification / end-to-end ISP + perception | RAW sensor data | 真实与模拟 | 不依赖 RGB foundation model | 端到端训练 | Input-level + end-to-end | 联合优化 demosaic / denoise / tone map 与 perception task | 是早期 task-driven ISP 代表作 | 更像“替换 ISP 并端到端重学”，不适合直接套在 frozen DAV2 上 |
| DynamicISP citeturn22search5turn22search1 | 2023 | ICCV | Image recognition / detection | RAW through dynamically controlled ISP | 真实 RAW 任务 setting | 面向 recognition backbone | 可训练 ISP + downstream model | Input-level | 动态控制经典 ISP 参数以服务 recognition | 说明 ISP 不是固定最优，任务相关 ISP 有价值 | 依然是 pre-processing 优化，不解决 depth foundation prior 被破坏的问题 |
| InvISP citeturn37search0turn37search4turn37search12 | 2021 | CVPR | Forward / inverse ISP | sRGB ↔ RAW | 真实 DSLR 配对数据训练 | 不适用 | N/A | Representation / synthesis | 可逆 ISP，同步支持 forward rendering 与 RAW reconstruction | 你当前 synthetic RAW 数据生成的关键参考 | 不是 depth；也不能把反推 RAW 解释为恢复了真实缺失传感器信息 |
| ParamISP citeturn37search1turn37search13 | 2024 | CVPR | Forward / inverse ISP | sRGB ↔ RAW + camera parameters | 真实 paired ISP modeling | 不适用 | N/A | Representation / synthesis | 用相机参数控制 forward / inverse ISP | 对“更物理真实的 synthetic RAW”有帮助 | 仍然不等于真实 RAW 采样；VKITTI 没有真实 EXIF / sensor process |
| CycleISP citeturn37search2turn37search10 | 2020 | CVPR | Real image restoration / improved synthesis | RAW + sRGB | 学习 forward / reverse pipeline 生成 realistic pairs | 不适用 | N/A | Representation / synthesis | 通过 RGB2RAW + RAW2RGB 建 realistic noisy pairs | 支持你比较不同 unprocessing pipeline | 面向 restoration，不直接回答 depth |
| Learning to See in the Dark / SID citeturn38search0turn38search4 | 2018 | CVPR | Low-light RAW enhancement | RAW sensor data | 真实 RAW 低照 paired data | 否 | 端到端训练 | Input-level | 直接在 RAW 域处理极暗场景，建立 SID dataset | 是 RAW 低照视觉的奠基工作 | 不涉及 depth，也不处理 foundation model 适配 |
| Self-Supervised Monocular Depth Estimation in the Dark citeturn32view0 | 2024 | IJCAI | Night monocular depth | Day RGB + inverse-ISP / noise simulation for training | 训练中构造 simulated dark raw-like process | 基于 RGB depth backbone | 训练 backbone | Training / augmentation | 用 imaging physics 与 inverse-ISP-inspired noise synthesis，提升夜间 MDE 泛化 | 是与你最接近的 depth 方向间接证据 | 不是 test-time RAW branch，也不是 frozen foundation correction |
| Depth Prompting for Sensor-Agnostic Depth Estimation citeturn35view0 | 2024 | CVPR | Depth estimation / completion | RGB + sparse depth prompt | 非 RAW | 是，foundation monocular depth model | 仅 tuning 约 0.1% 参量，其余冻结 | Feature + refinement | 用 depth prompt 嵌入 foundation model，修正传感器偏置并输出 metric depth | 强烈支持“冻结主模型 + 小模块适配新传感器输入” | 辅助模态是 sparse depth，信息量远强于 synthetic RAW-like |
| PromptDA citeturn12search1turn12search5turn12search8 | 2025 | CVPR | Metric depth | Depth Anything + low-cost LiDAR prompt | 非 RAW | 是，Depth Anything | decoder 级 prompt fusion | Feature / decoder | 用低成本 LiDAR prompt 引导 Depth Anything 输出高分辨率 metric depth | 证明强 RGB depth foundation 可被外部模态“提示式修正” | 辅助信号是几何测量，不是表征重参数化 |
| Polarization Prompt Fusion Tuning / PPFT citeturn33view3turn34view3 | 2024 | CVPR | Depth enhancement | Sensor depth + RGB/intensity + polarization | 非 RAW | 是，CompletionFormer / RGB-based foundation | 预训练 backbone 上加入新 prompt fusion path | Feature-level | 用 polarization 作为 prompt 修正深度，特别是透明面与形状错误 | 最像你想做的“辅助模态只做困难区域补偿” | 目标是 depth enhancement，不是 monocular depth foundation residual correction |
| Depth Any Event Stream / EventDAM citeturn33view0 | 2025 | ICCV | Event-based monocular depth | Event + paired RGB during distillation | 非 RAW | 是，Depth Anything Model | 教师固定，学生适配 | Distillation + feature | 通过 sparsity-aware distillation 把 RGB depth foundation 的知识迁到 event | 说明直接塞新模态不行，需要受控 distillation / mixture | 目标模态与 RAW 差异比 RAW-sRGB 更大，不能直接照搬 |
| Depth AnyEvent citeturn33view1 | 2025 | ICCV | Event-based monocular depth | Event | 非 RAW | 是，image-based VFMs | 适配 image-based VFM 到 event 域 | Distillation / adaptation | 用 RGB-VFM 生成 proxy labels 与适配策略进入 event 域 | 支持“teacher-student + cross-modal adaptation” | 不是同步 RGB 主路径 + auxiliary correction |

## Directly related works

### RAW beyond RGB 的基础证据

RAW-Adapter、Beyond RGB、AODRaw、Dirty Pixels、DynamicISP 这一脉络，已经相当稳定地支持一个结论：**当输入是真实 camera RAW 时，传统 ISP 为人眼优化，并不一定对机器感知最优**。RAW-Adapter 明确把问题表述为“如何把 information-rich RAW 适配到 knowledge-rich sRGB pre-trained model”，并指出训练从头学 RAW 会显著伤害高层任务性能；Beyond RGB / RAM 进一步把传统顺序式 ISP 改成并行任务驱动前端，在七个公开 RAW detection 数据集上都获得提升；AODRaw 则从 benchmark 角度再次强调，sRGB pretraining 与 RAW domain 之间存在显著 gap，而 neural ISP 虽能缩小差距，却也带来额外成本。citeturn24view0turn24view1turn29view0turn30view0turn31view0

这条证据链对你的启发是**正面的，但不是直接可移植的**。原因在于 detection / segmentation 的 loss 对输入表征的容忍度更高，模型只需要在候选区域内保持足够的物体语义与部分边缘即可，而 monocular depth 尤其是 foundation depth 的目标是**整图稠密、全局一致、局部边界锐利、语义与几何同时稳定**。换句话说，RAM 一类前端即便对 detector 有利，也可能改变 DAV2 赖以建立深度先验的 shading、semantic layout、texture ordering 与 sky/far pri纳，因此你看到 “loss 下降但边界和天空变糊” 并不反常，反而符合 dense geometry task 对输入分布更敏感的预期。这一点虽然没有被某一篇论文直接写成“RAW front-end 伤害 depth boundary”，但 RAW-Adapter、AODRaw 对 domain gap 的观察，加上 PPFT / EventDAM 对模态注入非平凡性的强调，已经给出很强的间接支持。citeturn24view0turn31view0turn34view3turn33view0

Dirty Pixels 和 DynamicISP 还带来另一层启发：**task-driven ISP 在原则上是成立的，但它更适合“任务定义本身就是输入表征学习”的场景**。Dirty Pixels 联合优化低层处理与分类，在低照等条件下有效；DynamicISP 则动态调节经典 ISP 参数服务 recognition。你的情况不同，因为 DAV2 本身已经是一种超强的 RGB depth prior，真正稀缺的不是重学一个输入管线，而是**如何在不伤主模型的情况下，把新的补充 cue 注入进去**。这正是为什么你的研究问题应该主动从 “RAW replace RGB” 转向 “RAW-like cues correct failure regions of a frozen RGB depth FM”。citeturn26search0turn26search2turn22search5turn24view0

### Inverse ISP 与 synthetic RAW 的边界

InvISP、ParamISP、Unprocessing、CycleISP 证明了两个事实。第一，**可以**从 sRGB 反推或合成一个较真实的 RAW-like 表征，且这种表征对某些视觉任务或数据合成工作有实际价值；第二，**不应该**把这种表征混同为“恢复了真实传感器里原本存在、但 sRGB 中已经消失的全部信息”。InvISP 的表述是“recover nearly perfect RAW data” 和 “reconstruct realistic RAW data”，这是在其可逆建模与特定 DSLR 配对数据条件下成立的重建目标；Brooks 的 unprocessing 更直白，是为了“synthesize realistic raw sensor measurements from Internet photos”；CycleISP 则强调 forward + reverse pipeline 用于生成更 realistic 的 RAW / sRGB noisy pairs；ParamISP又补充了 camera parameters 对 forward / inverse ISP 的重要性。它们共同支持你“可以做 RAW-like 表征”这一做法，但**不支持**“从 VKITTI 的 sRGB 反推后得到的就是比原图多出来的真实 sensor information”。citeturn37search0turn37search4turn37search1turn37search13turn23search0turn23search1turn37search2turn37search10

所以，在你这篇论文里，关于 synthetic RAW 最稳妥的三条表述边界是这样的。第一，不说 “RAW provides additional sensor information”，改说 “**inverse-ISP-generated RAW-like representations offer a sensor-inspired alternative parameterization**”。第二，不说 “recover lost information”，改说 “**expose linearized intensity relationships, transformed tone statistics, and alternative cues that may be easier for a correction branch to exploit**”。第三，不把 VKITTI 结果写成对 real RAW 的定论，而写成 “**a proof of concept for representation-level refinement under controlled alignment**”。这样 reviewer 即使追问“你的 RAW 是从 RGB 反推的，怎么会比 RGB 信息更多”，你的回答依然是闭环的：**我们的 claim 不是恢复真实丢失信息，而是测试一种 sensor-inspired complement representation 是否能帮助 frozen depth prior 做局部修正**。citeturn37search0turn23search0turn23search1turn37search13

### RAW for depth 或 sensor-level for depth

这部分的核心结论必须说得很清楚：**在本次检索覆盖的主流 CV 论文中，我没有找到已经成为共识或代表性的“RAW helps monocular depth estimation”论文，更没有找到与你完全同构的“frozen DAV2 + RAW residual correction branch”工作。**最接近的直接项有两个，但都不够强。其一是一个 2025 arXiv 的 Bayer-domain video CV 框架，它把 NYU Depth V2 先通过 InvISP 转成 Bayer，再做 monocular depth 预测，不过它的关注点是**Bayer 域高效推理与 motion estimation acceleration**，不是 foundation model adaptation，也不是 RAW 作为 complementary cue；其二是一个 2025 博士论文级工作的 “BayerDepth”，但这不是主流同行评审会议论文，不能作为你 related work 的主要对手。citeturn21view0turn20search4

真正更有价值的，是几类**间接但高度相关的 depth 证据**。Self-Supervised Monocular Depth Estimation in the Dark 并没有在测试时用 RAW 作辅助模态，但它明确利用了 inverse-ISP / raw noise physics 来补偿夜间数据分布，说明**成像链建模对 MDE 的确可能有帮助**，尤其在低照和噪声条件下；不过它的帮助方式是 training-time compensation，而不是 inference-time RAW branch。EventDAM 与 Depth AnyEvent 则把 RGB depth foundation model 的知识迁到 event 域，强调直接把另一种感知信号塞进 image-based FM 会受到 density discrepancy 和 modality mismatch 限制，因此要靠 sparsity-aware feature mixture、distillation 和 consistency 这类受控机制。PPFT 则在深度增强任务中展示了**辅助模态不是替代 RGB，而是作为 prompt 去修正透明物体、孔洞和形状错误**，这和你想让 RAW branch 只修 difficult regions 的想法在方法哲学上最接近。citeturn32view0turn33view0turn33view1turn33view3

换句话说，你的 gap 不是“第一个把 RAW 用到 depth 的人”，而是更具体的：**第一批把 RAW-like representation 定位为 frozen RGB depth foundation model 的 complementary residual cue，而不是主输入替代物的人之一。**这个 gap 的成立，依赖你把 claim 控制在 representation-complementarity 和 failure-region correction 上，而不是 sensor-information superiority 上。citeturn24view0turn29view0turn35view0turn33view3

### Depth foundation model adaptation 的直接启发

Depth Anything 与 DAV2 把 monocular depth foundation model 推到了一个很高的强度，前者通过大规模 unlabeled + labeled 数据形成强零样本泛化，后者进一步用合成标注数据替换真实标注、扩大 teacher 容量、桥接大规模 pseudo-labeled real images，并显著增强了细节和鲁棒性。MiDaS 奠定了 affine-invariant relative depth 的多数据集混训范式；ZoeDepth 表示可以在强 relative prior 之上通过 lightweight metric heads 获得 metric transfer；Metric3D 与 UniDepth 进一步面向 zero-shot metric 3D / metric depth 做 foundation-style 建模；Marigold 则展示了 generative foundation prior 也可以被 repurpose 到 monocular depth。这个谱系的共同含义是：**强基础模型的 prior 很值钱，适配时不应轻易破坏输入分布和主干特征流。**citeturn36search2turn36search10turn9search12turn9search16turn9search13turn36search3turn36search17turn9search15turn36search20

在“如何做轻量适配”这件事上，最直接支持你的是 Depth Prompting 与 PromptDA。Depth Prompting 把 sparse depth 作为 prompt 嵌入到预训练 monocular depth foundation model，只 tuning 约 0.1% 参数，其余保持冻结，并强调 joint representation 对 sensor bias 很敏感，因此要采用 disentangled prompt engineering；PromptDA 则直接把 low-cost LiDAR prompt 多尺度融合到 Depth Anything decoder 中，把 foundation model 变成“局部形状学习器 + metric prompt 修正器”。这两篇都在告诉你一件事：**强 depth foundation model 完全可以作为被冻结的主先验，然后用一个小而专门的分支去改它。**citeturn35view0turn12search1turn12search5turn12search8

另外，DepthAnything-AC 这类工作也说明，基础深度模型在复杂条件下会出问题，但可以通过**小数据、无标注 consistency-based fine-tuning** 来增强边界和细节表现；PPEA-Depth 则明确指出，在小数据上直接 fine-tune 预训练深度模型容易破坏其已学到的 generalized patterns，因此参数高效适配通常更稳。虽然它们不是 RAW 论文，但它们都支持你的策略选择：**如果目标是保住 DAV2 的 RGB prior，优先考虑 frozen main path + trainable correction branch，而不是对 DAV2 主干做大幅更新。**citeturn12search0turn12search7turn10search7

## Gap analysis and judgement

### 你的想法与已有工作的差异

先直接回答最关键的问题：**我没有找到一篇已发表工作明确提出“frozen RGB depth foundation model + RAW or synthetic RAW residual correction branch”**。RAW-Adapter 是“把 RAW 输入适配到预训练视觉模型”；RAM / Beyond RGB 是“用 RAW-driven pre-processing 替代或重塑传统 ISP，再供 detector 使用”；AODRaw 是“说明 sRGB pretrain 到 RAW 有 gap，并转向 RAW pretraining + distillation”；Depth Prompting、PromptDA、PPFT、EventDAM 则分别说明“冻结/轻调 foundation model + 新模态 prompt / distillation / fusion”是合理的，但这些新模态不是 RAW-like。把这两条线真正交叉起来，就是你的创新空间。citeturn24view0turn29view0turn31view0turn35view0turn12search5turn33view3turn33view0

因此，我建议把 novelty 放在四点，而不是分散。第一，**preserve RGB depth prior**：不让 RAW-like 破坏 DAV2 主路径。第二，**RAW-like as complementary residual cue**：RAW-like 不替代深度预测，只做 error-aware correction。第三，**avoid input-distribution shift**：拒绝 RAW-to-3ch 直接喂入 foundation model，而在 output / decoder 层做受控修正。第四，**systematic analysis of when RAW-like helps and when it hurts**：特别是对 boundary、dark / saturated 区域、小目标、transparent / reflective hints、以及 sky / far region 的分区域分析。前两点是 formulation novelty，后两点是 paper-level positioning novelty。citeturn24view1turn31view0turn35view0turn33view3

### 你的已有失败实验为什么合理

你报告的失败现象里，我认为最可能的解释依次是下面五条。

最可能的是 **输入分布失配**。AODRaw 直接报告了 sRGB-pretrained detector 在 RAW 域 fine-tune 会受 domain gap 限制，甚至不如 sRGB-based baseline；RAW-Adapter 也显示使用 sRGB pretrain 很关键，但 RAW 仍需要专门适配器。这对 DAV2 只会更严重，因为深度是 dense continuous prediction，对输入分布的鲁棒性通常比 detection 更脆弱。citeturn31view0turn24view0turn24view1

第二个高概率原因是 **你在重写 DAV2 依赖的语义-几何映射**。Depth foundation model 的强项不只是纹理边缘，更是 object prior、scene layout prior、sky-ground relations、far-region semantics。把 RAW-like 表征压成 3-channel 再喂进去，本质上是在让一个 “为 RGB 统计训练出来的 feature extractor” 面对另一种图像统计。EventDAM 明确说 dense RGB 与 sparse events 的差异会阻止 DAM 直接工作；PPFT 也明确说，把另一种模态接到预训练基础模型上并不 trivial，需要专门 block 缓解 modality misalignment。RAW-like 与 RGB 的差异没有 event 那么极端，但对 DAV2 来说已经足够大。citeturn33view0turn34view3

第三个原因是 **synthetic RAW 的额外信息并不真实存在**。InvISP / Unprocessing 能构造 RAW-like 域，但它们并不能从已经被 tone mapping、gamma、compression、quantization 处理过的 sRGB 中“凭空恢复”真实丢掉的 sensor information。因此 RAW-front-end 很可能学到的是一种数据集特定的重参数化 shortcut，而不是普适 sensor cue。这个解释与你观察到“global scale 还行，但边界和天空受损”是一致的，因为 scale 往往可以靠数据集偏置学到，而局部结构和语义边界则更依赖 foundation prior。citeturn37search0turn23search0turn37search2

第四个原因是 **边界与困难区域本来就需要专门 refinement 机制**。Mind The Edge 与 Predicting Sharp and Accurate Occlusion Boundaries 都说明，深度边界不是普通回归 loss 自然会学好的东西，必须有 edge-aware / boundary-aware 的专门设计。你的 RAW front-end 没有内建“只修边界”的约束，它很容易为了整体 loss 而牺牲局部锐度。citeturn15search15turn15search4

第五个原因是 **小数据下重新适配 foundation model 风险高**。PPEA-Depth 明说小数据直接 fine-tune 可能 disrupt generalized patterns；Depth Prompting 则通过只调 0.1% 参数来避免这一点。对于 VKITTI 这种规模和场景多样性都不如 foundation pretraining corpora 的数据集，这个风险很现实。citeturn10search7turn35view0

这些解释都可以被实验验证。最关键的诊断实验分别是：一，比较 RGB-only residual、RAW-only residual、RGB+RAW residual 的增益差异，用来判断 RAW-like 是否真的提供增量；二，比较 output-level residual 和 input replacement 对边界指标、sky/far 指标、small-object 指标的影响；三，比较 InvISP 与 generic unprocessing 的差异，用来判断收益是否只是某种 representation artifact；四，做 DAV2 的 TTA/ensemble uncertainty 可视化，看 RAW branch 的 gating 是否真的集中在高不确定区域；五，直接可视化 residual 和 gate map，检查它是不是全图开启。citeturn35view0turn33view4turn15search15

### synthetic RAW 的问题该怎么回答 reviewer

你提出的十个问题里，最重要的是前四个，我给出明确结论。

**第一，synthetic RAW / inverse-ISP RAW 不能单独支持“RAW contains information beyond sRGB”这个 claim。**因为在你的实验里，RAW-like 是由 sRGB 反推出来的，它没有经历真实 sensor acquisition，也不可能恢复真实 ISP 已丢失的所有物理信息。InvISP 和 ParamISP 的成功说明的是“可重建 / 可近似 / 可利用”，不是“超越信息守恒”。citeturn37search0turn37search4turn37search13

**第二，你应该把 claim 改写成 representation-level claim。**最好的写法是：
“**inverse-ISP-generated RAW-like representations provide a sensor-inspired alternative parameterization that can expose complementary cues for refining a frozen RGB depth foundation model**.”
如果你想再保守一点，可以把 “expose” 改为 “re-weight / re-express”。这比 “provide additional sensor information” 安全得多。citeturn23search0turn37search0turn37search13

**第三，VKITTI + synthetic RAW 完全可以作为第一阶段 proof of concept，但只能证明方法学，不证明传感器信息论。**它非常适合回答“补充式 correction 比替代式输入更稳吗”“gate / uncertainty / residual formulation 是否有效”“在严格像素对齐条件下，RAW-like branch 是否能找到 DAV2 的 failure regions”。但它不适合支撑强现实 claim。citeturn24view0turn25view0turn37search0turn23search0

**第四，如果要把 claim 做强，需要补上真实 RAW-depth 数据，或者更物理真实的 rendering pipeline。**本次检索里我没有找到一个已经被 monocular depth 社区广泛采用的“paired real camera RAW + dense depth ground truth”公开 benchmark；相关公开 RAW 数据更多集中在 detection、low-light restoration，而公开 depth benchmark 大多是 RGB+depth。基于这一现实，你可以把论文定位为 synthetic RAW-like proof-of-concept，并在 discussion 里明确 future work 是补充 real RAW-RGB-depth 或 renderer-native linear sensor output。这个表述比勉强把 VKITTI 说成 real RAW proxy 要诚实得多，也更不容易被审稿人抓住。citeturn24view0turn31view0turn23search6turn18search6turn19search17

为了防 reviewer，我建议采用五种回应策略同时部署。

**方法定位策略**：强调你研究的是 “complementary correction under frozen RGB depth prior”，不是 “real RAW superiority”。

**实验补强策略**：加入 RGB-only residual control、parameter-matched control、InvISP vs unprocessing、global-only vs local-only correction、gate sparsity ablation，证明收益不是简单增加参数。相关工作如 RAM、RAW-Adapter、Depth Prompting、PPFT 都在 ablation 里强调 architecture choice 而不只是参数量。citeturn30view0turn25view2turn35view0turn34view3

**数据集补强策略**：如果短期拿不到 real RAW-depth，就增加更物理真实的合成实验，例如对 VKITTI 的 RGB 先做 controlled over/under-exposure、gamma perturbation、noise perturbation，再比较 RGB main path 与 RAW-like correction 的鲁棒性，这样你证明的是 representation robustness，而不是 sensor information。

**claim 降级策略**：把标题、摘要、贡献都写成 RAW-like / inverse-ISP-inspired / sensor-inspired，不写 sensor RAW superiority。

**ablation 设计策略**：做 “RAW-like 对哪些区域有效，哪些无效” 的 failure-region analysis。如果结果显示它只帮助 dark / saturated / small-object 边缘区域，而对 sky / far semantic-heavy 区域帮助不大，这恰恰是可信结果，不是坏结果。PPFT 在透明面和形状修正上的成功，也说明辅助模态通常不是全能，只在 RGB 歧义最强处最有价值。citeturn33view3

## Recommended method direction

### 优先级判断

如果你只做一条主线，我建议你把方法收敛到一个**A + B + D 的混合体**，也就是：

**frozen DAV2 RGB main path + RAW-like error/confidence predictor + gated residual / affine correction head**

这是当前设定下最有希望的 formulation。它比纯 A 更稳，因为它不强迫 RAW branch 到处输出残差；比纯 B 更具体，因为 uncertainty 最终要落实成如何修正；比纯 D 更强，因为只预测 error map 容易过弱，最好让它顺带输出 correction magnitude。它也最符合 synthetic RAW 的 claim 边界，因为你可以把 RAW-like 解释为一种“帮助定位与修正 RGB depth failure”的辅助表示，而不是完整几何模态。这个定位，与 Depth Prompting 的 disentangled adaptation、PPFT 的 prompt fusion、CaFNet 的 confidence-aware gated fusion、EventDAM 的受控 cross-modal transfer 在方法精神上是同向的。citeturn35view0turn33view3turn33view4turn33view0

第二优先级是一个更强但更复杂的 **C 变体**：

**frozen DAV2 feature extractor + decoder-level RAW-like correction features + tiny refinement decoder**

只有在 output-level correction 明显 underfit 时，再上这个版本。因为 C 更容易被 reviewer 质疑“你只是多加了参数”，也更容易在小数据上破坏 DAV2 decoder feature statistics。除非你能用 parameter-matched control 和 clear ablation 证明“相同参数量下，RAW-like feature injection 明显优于 RGB-only refinement”，否则我不建议把 C 当第一主方法。citeturn35view0turn34view3

下面给出五个 formulation，但我会明确标出主推与备选。

### Formulation A

**Overall pipeline.** sRGB 输入 frozen DAV2，输出基础预测 \(Y_{rgb}\)。这里如果用 DAV2 relative model，我建议把 \(Y\) 定义成**affine-invariant disparity / inverse-depth-like output**，而不是 metric depth。synthetic RAW-like 输入一个轻量 U-Net / ConvNeXt-Tiny / MobileNet-style encoder，和 \(Y_{rgb}\)、RGB 边缘图、可选的 DAV2 中间 feature 一起进入 correction head，输出一个全局仿射项 \((s,t)\)、一个局部残差 \(\Delta Y\) 和一个 gating map \(g\)。最终 \(Y_{final}=s\cdot Y_{rgb}+t+g\cdot \Delta Y\)。如果是 metric depth 版本，则改用 log-depth：\(z=\log d\)，令 \(z_{final}=z_{rgb}+\alpha+g\cdot \Delta z\)，最后指数映射回正深度。这个设计与 ZoeDepth 的 relative-to-metric 头部思想在精神上相近，也与 PromptDA 中“主模型负责形状，外部提示负责尺度/细化”的分工一致。citeturn36search3turn12search5

**为什么我不建议 relative setting 直接在 linear depth 上做 residual。** DAV2 / MiDaS 一类 relative 模型本质上更接近 affine-invariant disparity；如果你直接在 linear depth 上做残差，训练时 GT 对齐与测试时未知尺度之间会产生额外不稳定。更稳的方法是：relative 版本在 \(Y\) 空间做 global affine + local residual；metric 版本再切到 log-depth residual。MiDaS 的 affine-invariant设定、ZoeDepth 的 metric bin / head 设计、Metric3D / UniDepth 的 metric建模都支持这种“先分清相对与绝对空间，再决定 correction parameterization”的思路。citeturn9search13turn36search3turn36search17turn9search15

**Trainable modules.** 只训练 RAW-like encoder、gating head、residual head、global affine head。DAV2 encoder 和 decoder 全冻。若需要额外稳态，可训练一个极小的 post-refinement decoder，但不要回传到 DAV2 主干。

**Losses.** Relative 版本使用 scale-and-shift invariant loss + gradient / edge loss + residual regularization + gate sparsity regularization + teacher consistency。Metric 版本使用 SILog / L1(log-depth) + gradient / boundary loss + residual / gate regularization。边界项的必要性，可参考 Mind The Edge 与 Occlusion Boundary Refinement 的结论。citeturn15search15turn15search4

**Regularization.**
其一，**identity initialization**：residual head 最后一层零初始化，gate bias 设为负值，让初始 \(g\approx 0\)。
其二，**gate sparsity**：\(\lambda_g\|g\|_1\) 或 \(\lambda_g \text{mean}(g)\)，避免全图开启。
其三，**teacher consistency**：在教师低误差区逼近 \(Y_{rgb}\)，避免坏改动。
其四，**residual magnitude penalty**：限制 \(\Delta Y\) 分布，鼓励小改而不是重写整图。
这套设计与 CaFNet 的 confidence-guided fusion、Depth Prompting 的 small tuning、PPFT 的 prompt-style injection 在“防止 auxiliary branch over-correct”上是一致的。citeturn33view4turn35view0turn34view3

**Expected advantage.** 最大优势是**稳定**。你不会改 DAV2 的输入分布，也不会让 RAW-like branch 背负完整 depth prediction，branch 只需学“如何改错”。这最适合小规模 VKITTI 和 synthetic RAW 的设定。

**Potential failure mode.** 如果 \(g\) 学成全图常开，方法会退化成一个 second-stage depth head，最终把 DAV2 prior 重新写掉；如果 branch 太弱，则只学到 trivial 零残差。这个 formulation 的关键不是网络本身，而是 regularization 是否把它锚在 “small but meaningful correction” 上。

**Novelty.** 中等偏强。模块不新，但 problem formulation 新，尤其是在 RAW-like + frozen depth FM + failure-region correction 这一交叉点上。

**Minimum viable experiment.** 只用 DAV2-relative + VKITTI + InvISP-RAW-like，预测 affine + gated residual，证明它优于 RAW-to-3ch 和 RAM-like front-end，并且 gate 主要激活在高误差区域。

**Necessary ablations.** 无 gate、无 affine、无 residual regularization、RGB-only branch、RAW-only branch、parameter-matched RGB refinement。没有这些 ablation，这个 formulation 很难说服 reviewer。

### Formulation B

**Overall pipeline.** 先运行 frozen DAV2 获得 \(Y_{rgb}\)，再估计一个 uncertainty / error-likelihood map \(u\)。RAW-like branch 只在高 \(u\) 区域触发 correction。最终可写成 \(Y_{final}=Y_{rgb}+m(u)\cdot \Delta Y\)，其中 \(m\) 是由 uncertainty 派生的门控。对应训练时，你既可以让 branch 直接预测 \(u\)，也可以用 DAV2 的 TTA / ensemble disagreement 作为软 target。EventDAM 的稀疏感知 distillation、CaFNet 的 confidence map 监督和 confidence-aware fusion，都说明“先估可信度，再做融合”是成熟范式。citeturn33view0turn33view4

**Uncertainty 怎么获得。** 我建议按照“从简到繁”的顺序做。第一版最简单的是**supervised error likelihood**：用 GT 定义教师误差 \(e_{rgb}\)，把 top-k error 区域或连续归一化误差作为 uncertainty target，让网络学预测它。第二版可以加 DAV2 的 TTA uncertainty，缓解 train-test mismatch。第三版才考虑 heteroscedastic NLL 形式的 aleatoric uncertainty。对于 VKITTI 这种全监督 synthetic 数据，**直接监督 error-likelihood 是最划算的**。citeturn33view4turn14search4turn13search19

**Train-test mismatch 是否严重。** 会有，但可控。因为你不是拿 GT uncertainty 在测试时直接用，而是训练一个“误差可能性预测器”。这与 CaFNet 用 confidence ground truth 监督 confidence map 的思路类似。关键是别把 uncertainty 只做成 binary hard mask，否则很容易退化成 trivial segmentation。更稳的是预测连续 \(u\in[0,1]\)，再用 sparsity + calibration loss 去约束。citeturn33view4

**适配你任务的优点。** 这个 formulation 特别适合你说的 “错误不一定只局限在一种区域”。因为它先识别错误可能，而不是先定义错误类型。它也与 synthetic RAW 的限制更匹配：即使 RAW-like 没有新传感器信息，它仍然可能把某些难区从另一个参数化角度“显露”出来，帮助 error localization。

**风险。** 最大风险是 \(u\) 预测器学成“边缘检测器”或“暗区检测器”，导致只在显著纹理处开门，而不是真正学会 DAV2 何时会错。所以必须做 calibration / reliability 分析，看 predicted \(u\) 与实际 error 的相关性，而不仅看最终 depth 指标。

### Formulation C

**Overall pipeline.** DAV2 全冻，只抽取 2 到 3 个 decoder 或 late-backbone feature。RAW-like encoder 产生对应尺度的 correction feature，通过 FiLM、轻量 cross-attention、或者 gated residual conv 注入一个新的 tiny refinement decoder。最终输出 refined depth。这个设计与 PromptDA 的 decoder prompt fusion、PPFT 的 sequential prompt fusion block 最相近。citeturn12search5turn34view3

**何时比 A 更强。** 当错误不是简单可由 output residual 写出来，而需要访问一些深层局部语义 feature 时，C 会更强。比如小物体边缘、局部 geometric discontinuity，或者反射区周边的语义-边界耦合错误。

**为什么我不把它列为第一推荐。** 因为它更容易破坏 DAV2 现有 feature statistics，也更难做 clean evidence。你要证明“RAW feature 有用”而不是“只是又训练了一个新 decoder”，实验负担比 A 大一截。对第一篇会议稿，我更愿意把 C 作为 second-stage extension。

### Formulation D

**Overall pipeline.** RAW-like branch **不直接预测 depth**，而预测一个 error map、boundary correction map、local scale map 或 signed-correction template。最简单的实现是分两头：一头输出 error likelihood \(q\)，一头输出 signed residual \(\Delta Y\)，最终 \(Y_{final}=Y_{rgb}+q\cdot \Delta Y\)。更细一点还可以拆成 global affine correction \((s,t)\) + local boundary residual。这个 formulation 的好处是**叙事非常强**：RAW-like 不是新的 depth modality，而是一个 “RGB depth error predictor”。这会显著降低 reviewer 对 synthetic RAW 的攻击面。citeturn33view4turn15search4turn15search15

**Error supervision 怎么定义。** Relative 版本建议用**aligned disparity absolute error** 或 **rank-aware error**；metric 版本建议用 **absolute relative error** 或 **log-depth error**。如果你希望强调边界，额外定义一个 boundary error target，例如 GT depth gradient 与 predicted depth gradient 的不一致区域。Mind The Edge 和 Occlusion Boundary 相关工作都说明，边界错误最好单独考察。citeturn15search15turn15search4

**为什么这个 formulation 可能比直接 residual 更稳定。** 因为它先回答 “where DAV2 is wrong”，再回答 “how to fix it”。在 synthetic RAW 设定下，这种两阶段语义更合理。RAW-like branch 只要比 RGB-only 更善于预测错误区域，就有价值，不必承担完整几何建模。

**我的判断。** 如果你想最大化论文的 “problem formulation novelty”，D 其实是最漂亮的；如果想最大化首次实现的成功率，A/D 合并最好。

### Formulation E

**Overall pipeline.** DAV2 作为 frozen RGB teacher，student 是 correction system。训练时在 teacher 低误差区域用 consistency / distillation，在 teacher 高误差区域更多依赖 GT。也可以做成两阶段：先拟合 teacher confidence，再学 correction。EventDAM、Depth AnyEvent、以及近两年的 thermal / event distillation 方向都支持这种“强 RGB teacher + 新模态 student”模式。citeturn33view0turn33view1turn16search12

**适不适合 VKITTI。** 适合，而且是 synthetic 数据的一个优势，因为你有全监督 GT，可以明确定义 teacher-good / teacher-bad 区域。训练目标可以是：
\(L = w_{good}L_{consistency}(Y_{final},Y_{rgb}) + w_{bad}L_{gt}(Y_{final},Y^*)\)，
其中 \(w_{good}\) 与 \(w_{bad}\) 可由 GT teacher error 动态生成。

**隐患。** teacher consistency 如果太强，会阻碍真正纠错。所以 consistency 只能在低误差区域高权重，不能全图同权重。

**是否推荐作为首发主方法。** 我更愿意把它作为 A/D 的训练策略，而不是单独的一条模型线。

## Minimum viable experiment

### 最小可行模型与 baseline 组合

如果你想最快得到一篇可写的结果，我建议把实验主线收缩成下面这一组。

**核心 baselines。**
原始 DAV2 on VKITTI RGB；DAV2 RGB fine-tuning；RAW-to-3-channel + DAV2；RAM-like front-end + DAV2；RAW-Adapter-like adaptor + DAV2；simple RGB+RAW concat；frozen DAV2 + output-level affine+residual correction；frozen DAV2 + uncertainty-guided correction；RGB-only residual branch control；RAW-only residual branch control；parameter-matched refinement control。这样已经足够支撑主叙事。RAW-Adapter、RAM、AODRaw、Depth Prompting 的经验都表明，control 设计比多堆模型更重要。citeturn24view0turn29view0turn31view0turn35view0

**我建议删掉或后置的 baseline。** feature-level correction 可以放第二阶段，不一定进入第一轮主表；如果时间紧，teacher-student variant 也先作为训练策略 ablation，而不是独立 baseline。第一篇稿子的主表最好围绕一个核心问题：**补充式 correction 是否比替代式输入更稳、更有效。**

### 评价指标设计

**全局指标。** Relative 设定下采用 scale-and-shift aligned AbsRel、RMSE、SILog、δ1；metric 设定下报告原生 AbsRel、RMSE、SILog、δ1，并单独报告是否使用 predicted global scale correction。Depth Anything / MiDaS / ZoeDepth 的范式差异决定了 relative 和 metric 不能混写，必须分开汇报。citeturn9search13turn36search3turn9search12

**区域指标。** 这是你的论文成败关键。至少做以下几项：
边界区域误差；small-object 区域误差；dark region 误差；high-light / saturation region 误差；sky / far region 误差；textureless region 误差。
如果 VKITTI 里有语义标签或你能生成 pseudo segmentation，就分 semantic categories 报告。没有也没关系，至少基于亮度、梯度、实例面积、距离分桶。Mind The Edge、PPFT、CaFNet 都证明，困难区域分析常常比单个 AbsRel 更有说服力。citeturn15search15turn33view3turn33view4

**可视化。** 强制展示四列图：RGB、DAV2 原图预测、你的 gate / error map、最终预测。再叠加 depth edge 对比和 residual heatmap。没有这些图，你的 “RAW-like branch 只在困难区域改” 无法被 reviewer 直接看到。

### Ablation 的优先级

**必须做。**
frozen DAV2 vs RGB fine-tuned DAV2；input replacement vs output correction；RGB-only vs RAW-like-only vs RGB+RAW-like correction；with / without gate；with / without teacher consistency；with / without residual regularization；InvISP vs generic unprocessing；unseen VKITTI scenes generalization。
这些 ablation 直接对应你的核心 claim。做不到这些，文章会显得像一个普通 fusion trick。citeturn23search0turn37search0turn35view0

**锦上添花。**
output-level vs feature-level；additive residual vs affine correction vs log-depth residual；with / without uncertainty supervision；train data size sensitivity；synthetic over-exposure / under-exposure robustness；TTA uncertainty。
这些对投稿质量有帮助，但不必全部在首稿就做满。

### 结果好与不好时分别怎么解释

如果结果**明显优于** RAW-to-3ch 与 RAM-like front-end，但仍**弱于 RGB fine-tuning**，这依然是可发表结果，因为它支持的命题是：**在冻结 foundation model 的约束下，RAW-like 更适合作为补充式校正，而不是主输入替代。** 这本身就是一个清晰结论。citeturn24view0turn29view0turn35view0

如果结果只在**dark / saturated / small-object boundary** 这些局部区域提升，而全局 AbsRel 提升一般，也不是坏事。你本来就不该许诺全局全场景统一提升；相反，这会让你的 claim 更可信。PPFT 对透明/难区的增益、CaFNet 对噪声雷达点的 selective trust，都是“局部困难条件改得多，全局平均未必暴涨”的典型。citeturn33view3turn33view4

如果结果**不好**，最合理的解释有两种。其一，synthetic RAW-like 并没有提供比 RGB-only branch 更强的增益，说明 representation hypothesis 在当前数据上不成立；其二，DAV2 的错误主要来自高层语义与场景先验失配，而不是 ISP / tone / noise 相关 cue 缺失。在第二种情况下，你可以把工作重心从 RAW-like 转成 “RGB-only error-aware refinement under frozen DAV2”，把 RAW-like 作为 supplementary observation。这样至少不会空手而归。

### 建议的 losses 与训练细节

**relative 版本。** 让 DAV2 输出保持在 affine-invariant disparity 空间。训练主损失用 scale-and-shift invariant depth loss；再加 gradient / edge-aware loss 保边界；再加 weighted teacher consistency 抑制坏改动；再加 \(L_1\) 或 Charbonnier residual penalty 与 gate sparsity penalty。别直接优化 AbsRel 本身，因为 relative setting 下 AbsRel 依赖对齐方式，训练不稳定。

**metric 版本。** 如果你后续切到 DAV2 metric model，最好在 log-depth 空间做 correction。主损失用 SILog + \(L_1(\log d)\)；边界项继续保留；global scale head 单独回归一个 \(\alpha\) 或 coarse scale map；local residual head 只补局部细节。PromptDA 和 ZoeDepth 都可视为“尺度问题单独建模，细节问题另行补偿”的正面例子。citeturn12search5turn36search3

**初始化与优化。** correction head 零初始化，gate 偏置设为小开口；stop-gradient 传给 \(Y_{rgb}\) 更稳；DAV2 的 norm 层全冻；学习率设小，最好比普通 decoder 训练再低一级；可以前 1 到 2 epoch 只训练 gate / affine head，再联合训练 residual head，但不是必须。数据量不大时，warm-up 很值得做。Depth Prompting 和 parameter-efficient adaptation 的经验都支持小步、轻调、冻结主干。citeturn35view0turn10search7

## Risk analysis, writing suggestions, reading priority and final judgement

### 风险与 reviewer concern

最大的风险不是模型，而是**claim 过强**。如果标题、摘要、引言中把 synthetic RAW 写成 real RAW proxy，或者说 “RAW provides additional sensor information beyond sRGB”，你会非常容易被击穿。第二个风险是 reviewer 觉得你只是“在 DAV2 后面又接了一个小 U-Net”。第三个风险是如果你的提升只对 VKITTI 内有效，审稿人会怀疑 dataset-specific shortcut。第四个风险是如果 gate map 不可解释、全图都开，方法看起来像普通 refinement，而不是你号称的 complementary correction。第五个风险是 sky / far / boundary artifacts 如果在可视化里很明显，即便数值涨了，也会削弱说服力。citeturn15search15turn24view0turn31view0

### 论文动机与贡献怎么写

**动机段落**建议这样写：
当前 depth foundation models 在 RGB 域具有极强先验，但对暗区、高光、局部边界和部分异常成像条件仍会失败；直接把 RAW-like 作为替代输入会引入 distribution shift，破坏 foundation prior；因此我们研究是否可以让一种 sensor-inspired alternative representation 只作为**补充式纠错信号**，在保持 frozen RGB depth prior 的同时，修正其失败区域。这个动机同时调用了 RAW literature 的优势叙事与 foundation model adaptation 的保守策略。citeturn24view0turn31view0turn35view0

**贡献段落**建议压缩为三点。
其一，首次系统研究（或至少 very early study）**RAW-like representation as a complementary residual cue for frozen monocular depth foundation models**。
其二，提出一个 **error-aware gated residual correction** 框架，在不改变 DAV2 输入分布和主路径的前提下利用 RAW-like 表征。
其三，在 VKITTI + inverse-ISP-controlled setting 下系统比较 input replacement、feature adaptation、output correction，并给出 failure-region analysis。
这样写既有方法也有 empirical study，不会把全部赌注押在模块新颖性上。

### 建议的标题与 claim 强弱

下面给你八个更稳妥的标题，按 claim 强弱从弱到强排序。

**Complementary RAW-like Cues for Refining Frozen Monocular Depth Foundation Models**
最稳妥。强调 RAW-like、complementary、refining、frozen。几乎没有 overclaim 风险。

**Preserving RGB Depth Priors with RAW-like Residual Correction**
也很稳，亮点是保先验。适合你现在的主线。

**Error-Aware RAW-like Refinement for Frozen RGB Depth Foundation Models**
比上一个更方法化，适合如果你做了 error map / gate 可视化。

**Gated RAW-like Residual Correction for Monocular Depth Foundation Models**
方法感更强，但默认你主方法是 gated residual。

**Synthetic RAW-like Representation for Monocular Depth Foundation Model Refinement**
把 synthetic 写进标题，claim 最诚实，但吸引力略弱。

**Sensor-Inspired Inverse-ISP Representations for Depth Foundation Model Refinement**
非常 reviewer-friendly，尤其适合你想把 InvISP 当关键 data construction。

**RAW-Guided Failure-Region Correction for Monocular Depth Estimation**
比上面强一点，因为用了 RAW-guided，若全文其实是 synthetic RAW-like，建议正文一定解释清楚。

**RAW-Guided Residual Correction for Frozen Monocular Depth Foundation Models**
最抓人，但如果全文只有 synthetic RAW，建议不要用这个作为首选标题，除非副标题或摘要第一句马上降级说明是 inverse-ISP-generated RAW-like representation。

### 必读与次读

**必读十篇。**
RAW-Adapter，读它是为了学“如何把 RAW 适配到 sRGB 预训练模型”和 synthetic RAW related work 的写法。citeturn24view0turn24view1
Beyond RGB，读它是为了理解 RAM 范式、parallel ISP、以及 frozen pretrained detector setting 为什么在 detection 中可行。citeturn29view0turn30view0
AODRaw，读它是为了直接拿到“sRGB pretraining 对 RAW 有 domain gap”的论据。citeturn31view0
InvISP，读它是为了写 synthetic RAW 的数据构造和 claim 边界。citeturn37search0turn37search4
Unprocessing Images for Learned Raw Denoising，读它是为了 reviewer 讨论里解释 “从 RGB 反推 RAW-like 的理论边界”。citeturn23search0turn23search1
Depth Prompting for Sensor-Agnostic Depth Estimation，读它是为了“冻结 foundation model + 小 prompt 模块”的最直接方法学参考。citeturn35view0
PromptDA，读它是为了理解强 depth foundation model 如何被外部模态引导输出 metric / refined depth。citeturn12search1turn12search5
PPFT，读它是为了学习“辅助模态只做深度增强难区修正”的叙事和结构。citeturn33view3turn34view3
EventDAM 或 Depth AnyEvent，读它们是为了理解跨模态 distillation 怎样避免直接输入分布冲突。citeturn33view0turn33view1
Mind The Edge，读它是为了设计边界指标与 boundary-aware loss。citeturn15search15

**次读十篇。**
Dirty Pixels。citeturn26search0turn26search2
DynamicISP。citeturn22search5
ParamISP。citeturn37search1turn37search13
CycleISP。citeturn37search2turn37search10
Learning to See in the Dark。citeturn38search0turn38search4
Self-Supervised Monocular Depth Estimation in the Dark。citeturn32view0
CaFNet。citeturn33view4
Predicting Sharp and Accurate Occlusion Boundaries in Monocular Depth Estimation。citeturn15search4
Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging。citeturn15search10
PPEA-Depth。citeturn10search7

**可选方向。**
如果你后续拿到真实 RAW-depth 数据，去补读 AODRaw 之后沿 RAW pretraining、RAW-domain distillation 与更物理真实的 synthetic rendering；如果你发现 auxiliary cue 更像 uncertainty 而不是 geometry，就多看 confidence propagation、uncertainty estimation 一脉；如果你发现 transparent / reflective 区域最有增益，就往 polarization / depth enhancement 社区靠。citeturn13search19turn14search4turn33view3

### Open questions and limitations

本次调研没有发现一个已被社区广泛采用的 real camera RAW + dense monocular depth benchmark，所以“如何把 proof of concept 从 synthetic RAW-like 推到 real RAW-depth”仍是开放问题。另一个开放问题是，**synthetic RAW-like 在常规日间场景中是否真的能提供超出 RGB-only residual branch 的稳定增益**。如果你的 ablation 最终显示 RAW-like 与 RGB-only 残差差异很小，那么论文主张必须进一步降级，从“RAW-like complementary cues”退到“sensor-inspired alternative representation for error-aware refinement”。这并不致命，但会影响题目与贡献写法。citeturn23search6turn24view0turn31view0

### Final judgement and scores

先给最直接的结论：

**基于已有文献和你的实验设定，“保留 sRGB-DAV2 主路径，使用轻量 synthetic RAW / RAW-like branch 做 residual / uncertainty-guided correction”是一个有潜力、且明显区别于 RAM / RAW-Adapter 的研究方向。**

它的潜力主要来自：
一，它避免了你已经观察到会出问题的 input-distribution shift；
二，它顺应了 depth foundation model adaptation 文献里“冻结主模型、用小模块做受控适配”的趋势；
三，它把 synthetic RAW 的风险从“信息增益”转移到了“表示补充与失败区域校正”这一更安全的问题上；
四，目前确实缺少与你同构的直接先行工作。citeturn24view0turn29view0turn35view0turn33view3turn33view0

下面给出评分。除 **Risk / Experimental burden / Reviewer risk** 外，分数越高越好；后三项分数越高表示风险或负担越大。

| 维度 | 评分 | 判断 |
|---|---:|---|
| Novelty | 3.5 / 5 | 精确组合设定有新意，但 synthetic RAW 会削弱“RAW”本体创新性 |
| Feasibility | 4.0 / 5 | 技术上可做，且与你已有失败经验高度互补 |
| Risk | 4.0 / 5 | 主要风险来自 synthetic RAW claim 与增益可能局部化 |
| Suitability for VKITTI | 4.5 / 5 | 对 proof of concept 很合适，因对齐严格、GT 完整、便于 error supervision |
| Suitability for real RAW-depth future extension | 3.0 / 5 | 框架能迁移，但数据瓶颈是真问题 |
| Conference paper potential | 3.5 / 5 | 更像“扎实 formulation + ablation + analysis”型论文，投稿前景中上 |
| Experimental burden | 3.5 / 5 | 中等偏高，关键在于 control 与 region-wise analysis，而不是大模型训练 |
| Reviewer risk | 4.0 / 5 | 若 claim 不降级、ablation 不充分，风险高；若定位稳妥，可明显下降 |

如果只用一句话概括我的建议，那就是：

**继续做，但不要再把 RAW-like 放在 DAV2 前面“替代 RGB”，而要把它变成一个解释清楚、受约束、只在 DAV2 失败处出手的 correction branch，并且从论文第一句开始就把 claim 锚定在 synthetic RAW-like representation，而不是 real sensor RAW superiority。**
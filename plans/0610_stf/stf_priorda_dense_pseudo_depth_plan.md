# STF Dense Pseudo Depth Label 生成方案：DAv2-L + Prior Depth Anything

更新时间：2026-06-11

## 0. 执行状态（Codex 更新）

更新时间：2026-06-10 21:45（Asia/Shanghai）

当前状态：**PriorDA hires-fine 全量 6216 已完成；已新增 STF raw → teacher_rgb 的 10 张暗/过曝样本 DAv2-L 对比预览。**

已完成：

1. 环境确认：
   - 使用默认 `dav3` conda 环境。
   - `torch=2.10.0+cu128`，`torch.version.cuda=12.8`，`torch.cuda.is_available()=True`。
   - PriorDA 本体已用 `pip install -e codex_tmp/Prior-Depth-Anything-review --no-deps` 安装，repo commit 为 `8c029cb`。
   - PriorDA 权重已由 HF cache 加载：
     - `depth_anything_v2_vitb.pth`
     - `prior_depth_anything_vitb_1_1.pth`

2. `torch_cluster` 处理：
   - PyG `torch_cluster` wheels for `torch-2.10.0+cu128/cu126/cpu` 均因本机 `GLIBC_2.31` 低于 wheel 要求的 `GLIBC_2.32` 无法 import。
   - 未改写 `torch`，未切换 conda 环境。
   - 对本地 editable PriorDA repo 做了最小兼容补丁：
     - `codex_tmp/Prior-Depth-Anything-review/prior_depth_anything/depth_completion.py`
     - `codex_tmp/Prior-Depth-Anything-review/prior_depth_anything/sparse_sampler.py`
   - 补丁内容：`torch_cluster` 改为可选依赖；不可用时使用 `scipy.spatial.cKDTree` 做 exact KNN fallback。
   - `from prior_depth_anything import PriorDepthAnything` 已验证通过，当前 `torch_cluster_available=False`，`knn_backend=scipy_cKDTree_fallback`。

3. 新增项目脚本：
   - `finetune_stf/scripts/generate_stf_priorda_dense_pseudo_depth.py`
   - 功能包括：
     - 读取现有 STF DAv2-L manifest；
     - 构造 sparse LiDAR prior；
     - 构造 DAv2-L geometric proxy；
     - 调用 PriorDA；
     - 保存 raw / LiDAR overwrite / gated overwrite；
     - 保存 sparse mask、geometric meta、priorda meta、qa meta、debug visualization；
     - 生成 preview contact sheet；
     - 子集输出路径必须包含 `smoke` / `debug` / `tmp` / `codex_smoke`；
     - 全量运行必须显式传 `--preview-approved`。

4. 新增 tmux 启动脚本：
   - Preview：
     - `finetune_stf/scripts/formal/0610_run_stf_priorda_preview10_queue.sh`
   - Full：
     - `finetune_stf/scripts/formal/0610_run_stf_priorda_full_queue.sh`
   - Full 脚本会拒绝在未设置 `PREVIEW_APPROVED=/path/to/debug_priorda_preview10_*` 时启动。

5. 验证结果：
   - `py_compile` 通过。
   - `--help` 通过。
   - 子集输出路径保护通过。
   - full-run guard 通过。
   - 1 样本完整 smoke 通过：
     - output root：`/tmp/codex_smoke_priorda_single_0610`
     - hold-out AbsRel：`0.04635316327644239`
     - 成功后该 `codex_smoke` 目录已删除。
   - 一次较早失败 smoke 已按规范保留：
     - `/tmp/stf_priorda_no_marker_test`
     - 失败原因：修复前 debug visualization 使用了不兼容的 `matplotlib.colormaps` 导入。

6. 10 样本 preview gate 已完成：
   - tmux session：`priorda_stf_preview10_0610_2142`（已结束）
   - log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_2142_priorda_stf_preview10.log`
   - output root：`/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_0610_2142`
   - contact sheet：`/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_0610_2142/preview10_contact_sheet.png`
   - summary：`/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_0610_2142/preview10_summary.json`
   - 结果：
     - `total_rows=10`
     - `ok_rows=10`
     - `failed_rows=0`
     - `mean_absrel_holdout_ok=0.0574607584297732`
     - per-sample hold-out AbsRel range：`0.03838857521571735` 到 `0.10904144613562215`
   - 快速查看 contact sheet：非空，RGB / sparse overlay / DAv2 geometric proxy / PriorDA completion / overwrite 版本均正常显示。

下一步：

1. 人工查看：
   - `/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_0610_2142/preview10_contact_sheet.png`
2. 若确认无明显结构崩坏、尺度崩坏、边界异常或 overwrite 点状突变，再启动全量：

```bash
PREVIEW_APPROVED=/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_0610_2142 \
  bash finetune_stf/scripts/formal/0610_run_stf_priorda_full_queue.sh
```

启动后脚本会报告：

```text
tmux attach -t <SESSION>
tail -f <LOG_PATH>
```

## 0.1 Fine-stage 模糊诊断与 hires-fine 结论（Codex 更新）

更新时间：2026-06-10 22:15（Asia/Shanghai）

**现象**：preview10（`debug_priorda_preview10_0610_2142`）里 `dense_depth_priorda_raw` 明显比输入的 DAv2-L geometric proxy 糊，物体边界被过度平滑。

**根因（已在源码层确认）**：PriorDA 内部分 coarse / fine 两段。

1. coarse 段：传入 `geometric` 时 [depth_completion.py:88-90](../../codex_tmp/Prior-Depth-Anything-review/prior_depth_anything/depth_completion.py#L88-L90) 直接 `depth2disparity(geometric)`，用全分辨率 1024×1920 做 KNN/global 对齐，**不降采样，结构无损**。
2. fine 段：[__init__.py](../../codex_tmp/Prior-Depth-Anything-review/prior_depth_anything/__init__.py) 原本把工作分辨率**硬编码成 `heit = 518`**，再走 fine ViT-B forward [dpt.py:183-198](../../codex_tmp/Prior-Depth-Anything-review/prior_depth_anything/depth_anything_v2/dpt.py#L183-L198)：RGB 与 condition 被降采样到短边 518 → ViT-B 在 ~37×69 patch 上重生成 depth → 最后 `F.interpolate` 放大约 2× 回 1024×1920。高频细节在这一压一放里丢失，这就是模糊来源。脚本与数据本身无问题。

**代码改动（可控、默认行为不变）**：在 fine-stage 分辨率处加了 env 开关 `PRIORDA_FINE_HEIT`：

```python
# prior_depth_anything/__init__.py，fine stage 设置 heit 处
_fine_heit_env = os.environ.get("PRIORDA_FINE_HEIT", "518").strip().lower()
if _fine_heit_env in ("", "518"):
    heit = 518                                  # 原始默认 = 518-fine
elif _fine_heit_env == "native":
    heit = sparse_depths.shape[-2] // 14 * 14   # 全分辨率 = hires-fine（1024→1022）
else:
    heit = int(_fine_heit_env) // 14 * 14
```

- 不设变量 → `heit=518`，与原版完全一致；
- `PRIORDA_FINE_HEIT=native` → fine 段在原生分辨率工作，最后回插值近 1:1，不再放大致糊。
- 这是 engineering-only 的分辨率参数（不改 label 的几何语义来源），但因为它实际影响 fine 输出锐度，正式 launch 时应在脚本里显式写出，不靠隐式默认。

**对比验证（同 10 个确定性样本）**：驱动脚本 [0610_run_stf_priorda_preview10_blurcmp.sh](../../finetune_stf/scripts/formal/0610_run_stf_priorda_preview10_blurcmp.sh)，顺序跑 coarse-only 与 hires-fine。

| 版本 | fine 分辨率 | holdout AbsRel | 锐度 | 状态 |
|---|---|---|---|---|
| 518-fine（原始） | 518 | 0.0575 | 最糊 | 10/10 ok |
| **hires-fine** | native(1022) | **0.0546** | 中偏锐，边界清晰很多 | 10/10 ok，无 OOM，~5s/样本 |
| coarse-only | 跳过 fine | null（QA 未测出） | 最锐 | 10/10 qa_failed* |

\* coarse 的 `qa_failed` 来自 **clip 前**模型输出在个别像素有 inf/负 disparity；保存的 `dense_depth_priorda_raw`（clip 到 [1,80] 后）实际无 nan/≤0。但 coarse 的 metric 精度目前没有数字背书。

产物路径（均保留）：

```text
coarse-only : /mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_coarse_0610_2201/
hires-fine  : /mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_hiresfine_0610_2201/
放大对比图  : .../debug_priorda_preview10_coarse_0610_2201/zoom_compare_2018-02-04_10-11-22_00200.png
              （四栏：DAv2 geo / 518-fine / coarse-only / hires-fine）
```

**结论与决定**：全量 6216 采用 **hires-fine（`PRIORDA_FINE_HEIT=native`）**。理由：既消除模糊、又保留 PriorDA 学习式 metric 融合（符合本计划"让 PriorDA 内部做 alignment/refinement"的初衷），10 样本 holdout AbsRel 反而略升。coarse-only 留作"最大化 DAv2 结构保真"的备选，但正式用前需补 clip-前 invalid 像素处理与 coarse QA。

**全量前待办**：

1. 把全量脚本 [0610_run_stf_priorda_full_queue.sh](../../finetune_stf/scripts/formal/0610_run_stf_priorda_full_queue.sh) 显式加 `PRIORDA_FINE_HEIT=native`；
2. 全量过程中持续盯 holdout AbsRel，确认 1022 推理期外推分辨率在 6216 上不退化；
3. `priorda_meta.json` 建议记录 `fine_heit` 实际取值，便于追溯。

## 0.2 hires-fine 全量生成启动状态（Codex 更新）

更新时间：2026-06-10 22:26（Asia/Shanghai）

已按 0.1 的决定继续执行：

1. 已更新 full 启动脚本：
   - [0610_run_stf_priorda_full_queue.sh](../../finetune_stf/scripts/formal/0610_run_stf_priorda_full_queue.sh)
   - 默认 `FINE_HEIT=native`；
   - tmux 内显式导出 `PRIORDA_FINE_HEIT=native`；
   - 全量输出目录名加入 `hiresfine`，避免与 518-fine 版本混淆。

2. 已更新生成脚本：
   - [generate_stf_priorda_dense_pseudo_depth.py](../../finetune_stf/scripts/generate_stf_priorda_dense_pseudo_depth.py)
   - `run_config.json` 记录 `priorda_fine_heit_env`；
   - 每个 `priorda_meta/*.json` 记录：
     - `priorda_fine_heit_env`
     - `priorda_fine_heit_resolved`
     - `priorda_fine_heit_mode`

3. hires-fine 1 样本 smoke 已通过：
   - `PRIORDA_FINE_HEIT=native`
   - `priorda_fine_heit_env=native`
   - `priorda_fine_heit_resolved=1022`
   - `priorda_fine_heit_mode=native`
   - smoke 输出目录 `/tmp/codex_smoke_priorda_hiresfine_meta_0610` 成功后已删除。

4. 全量 6216 已启动：
   - tmux session：`priorda_stf_0610_2225`
   - log：`/home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_2225_priorda_stf_generate.log`
   - output root：`/mnt/drive/3333_raw/seeing_through_fog/0610_2225_pseudo_depth_priorda_hiresfine_dav2l_geo_lidar_last_6216`
   - approved preview：`/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_hiresfine_0610_2201`
   - fine stage：`PRIORDA_FINE_HEIT=native`
   - 启动日志已确认进入样本处理，当前看到 `[4/6216]`。

监控命令：

```bash
tmux attach -t priorda_stf_0610_2225
tail -f /home/caq/6666_raw/dav2_raw_0603/finetune_stf/logs/0610_2225_priorda_stf_generate.log
```

后续检查：

1. 运行中：
   - 观察日志是否出现 `[ERROR]`、CUDA OOM、NaN/inf 异常；
   - 定期查看 `manifest.csv` 行数与 `preview/run_summary`；
   - 若单样本失败，脚本会继续写入 `status=failed` 并在 `qa_meta` 保留错误。
2. 运行完成后：
   - 检查 `manifest.csv` 是否有 6216 条样本记录；
   - 汇总 `status_counts`、holdout AbsRel / RMSE / delta1；
   - 抽查 `debug_vis/` 和高 AbsRel 样本；
   - 确认每个 `priorda_meta` 均记录 `native -> 1022`；
   - 若失败样本数量非零，先保留全部产物并定位失败原因，不删除正式输出。

### 0.2.1 运行中进度检查（Codex 更新）

更新时间：2026-06-10 22:30（Asia/Shanghai）

当前全量任务仍在运行：

- tmux session：`priorda_stf_0610_2225`
- 日志最新看到：`[126/6216]`
- `manifest.csv` 已写入样本数：`125`
- 当前状态统计：`ok=125`，`failed=0`
- 当前 hold-out AbsRel：
  - mean：`0.04923665376754983`
  - median：`0.040466520025097236`
  - max：`0.19805435714433284`
- 最近 `manifest.csv` 与 log 均在持续更新，任务未卡死。
- 当前按 wall-clock 粗估速度约 `2.2 s/sample`；剩余约 `6091` 个样本，预计还需约 `3.7` 小时。
- 预计完成时间：约 **2026-06-11 02:10-02:20（Asia/Shanghai）**。

### 0.2.2 运行中进度检查（Codex 更新）

更新时间：2026-06-10 23:12（Asia/Shanghai）

当前全量任务仍在运行：

- tmux session：`priorda_stf_0610_2225`
- 日志最新看到：`[1909/6216]`
- `manifest.csv` 已写入样本数：`1908`
- 当前状态统计：`ok=1905`，`low_quality=3`，`failed=0`
- 当前 hold-out AbsRel：
  - mean：`0.06662771216882041`
  - median：`0.05570078448276436`
  - max：`0.3912850140855396`
- 最近 100 个样本脚本内平均耗时约 `0.94 s/sample`；按 wall-clock 总体速度估算约 `1.47 s/sample`。
- 剩余样本数：`4308`
- 预计还需约 `1.8` 小时。
- 预计完成时间：约 **2026-06-11 00:55-01:05（Asia/Shanghai）**。

### 0.2.3 hires-fine 全量完成状态（Codex 更新）

更新时间：2026-06-11 01:00（Asia/Shanghai）

全量 6216 已跑完，tmux session `priorda_stf_0610_2225` 已退出。

输出目录：

```text
/mnt/drive/3333_raw/seeing_through_fog/0610_2225_pseudo_depth_priorda_hiresfine_dav2l_geo_lidar_last_6216
```

完成文件：

- `run_summary.json`
- `run_config.json`
- `manifest.csv`

summary：

```json
{
  "total_rows": 6216,
  "ok_rows": 6212,
  "failed_rows": 1,
  "status_counts": {
    "ok": 6212,
    "low_quality": 3,
    "failed": 1
  },
  "mean_absrel_holdout_ok": 0.0631858433511088
}
```

输出文件计数：

| 子目录 | 文件数 |
|---|---:|
| `dense_depth_priorda_raw` | 6215 |
| `dense_depth_priorda_lidar_overwrite` | 6215 |
| `dense_depth_priorda_lidar_gated_overwrite` | 6215 |
| `dav2_geometric_proxy` | 6216 |
| `sparse_lidar_prior` | 6216 |
| `sparse_lidar_mask` | 6216 |
| `geometric_meta` | 6216 |
| `priorda_meta` | 6215 |
| `qa_meta` | 6216 |
| `debug_vis` | 6215 |

异常样本：

1. `failed=1`
   - `2018-02-04_21-48-59_00000`
   - split：`test`
   - 原因：`too few in-range sparse points for PriorDA: 0`
   - 该样本没有生成 dense PriorDA label；但 `geometric_meta`、`sparse_lidar_prior/mask`、`qa_meta` 已写入。

2. `low_quality=3`
   - `2018-02-06_16-09-22_00000`：in-range sparse points = `15`
   - `2018-02-09_11-51-53_00000`：in-range sparse points = `10`
   - `2018-02-09_19-10-30_00000`：in-range sparse points = `15`
   - 原因：sparse 点太少，无法做 hold-out QA；dense label 已生成，但质量标记为 `low_quality`。

AbsRel 分布（`ok` 样本，n=6212）：

- mean：`0.0631858433511088`
- median：`0.052996698053984`
- min：`0.010113338952599473`
- max：`0.3912850140855396`
- p90：`0.10809565276237917`
- p95：`0.1341817891086629`
- p99：`0.2257031473473903`

已抽查 `priorda_meta`：记录了 `priorda_fine_heit_env=native`、`priorda_fine_heit_resolved=1022`、`priorda_fine_heit_mode=native`。

后续建议：

1. 生成一个正式可用 manifest，默认排除 `failed`，是否排除 `low_quality` 需要按训练策略决定；
2. 抽查 high AbsRel top 样本的 `debug_vis`；
3. 如果训练集不需要 test split，可直接忽略唯一 failed 样本；如果后续流程要求 6216 全覆盖，需要为该样本设计 fallback（例如仅保存 DAv2 geometric proxy 或标记 skip）。

### 0.3 STF raw → teacher_rgb 的 DAv2-L 预览对比（Codex 更新）

更新时间：2026-06-11 02:08（Asia/Shanghai）

目的：效仿 ROD `teacher_bright_degreen_v1`，在 STF 上从 rectified RAW 生成偏亮 `teacher_rgb`，比较：

1. dataset 官方 RGB 输入的 DAv2-L prediction；
2. raw teacher_rgb 输入的 DAv2-L prediction；
3. sparse LiDAR GT（点加粗，不叠 RGB 背景）。

新增脚本：

- [make_stf_teacher_rgb_dav2_dark_overexp_compare.py](../../finetune_stf/scripts/make_stf_teacher_rgb_dav2_dark_overexp_compare.py)

关键设置：

- STF raw 读取：`cam_stereo_left_bayer_rect/npz/*.npz`
- raw key：`bayer_rect`
- raw channel order：`[R, Gr, Gb, B]`（按 STF rectified RAW README）
- raw → RGB：`[R, 0.5*(Gr+Gb), B]`
- teacher 配方：`gains=[1.08,0.95,1.10]`、`white_percentile=99.5`、`gamma=0.454545`
- raw normalization：`companded`（直接按 README 中 `0..3967 / 3967` 做 pseudo-RGB，再套 ROD teacher 亮化配方）
- DAv2-L checkpoint：`/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth`
- DAv2-L input size：`924`
- 两路 prediction 都由同一脚本重新跑同一个 DAv2-L，保证除了输入图像外 checkpoint / input size / evaluation protocol 一致。

筛选策略：

- 全量扫描 6216 张 STF 官方 RGB 的降采样亮度统计；
- 选 5 张最低 `rgb_luma_mean` 作为 `dark`；
- 选 5 张最高饱和/高亮比例作为 `overexposed`；
- 要求 sparse LiDAR in-range 点数不少于 `100`。

评估口径：

- 对每张图使用 deterministic sparse LiDAR holdout；
- prior points：拟合 DAv2 relative inverse prediction 到 `1 / depth` 的 affine；
- holdout points：计算 `AbsRel` 和 `D1(delta1)`；
- depth range：`1m..80m`。

输出目录：

```text
/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_teacher_rgb_dav2_compare
```

主要产物：

- contact sheet：
  - `stf_teacher_rgb_dav2_dark_overexp_10_spectral_r.png`
- metrics CSV：
  - `selected_dark_overexp_10_metrics.csv`
- summary JSON：
  - `summary.json`
- 中间结果：
  - `teacher_rgb/*.png`
  - `official_rgb_dav2l_pred/*.npy`
  - `teacher_rgb_dav2l_pred/*.npy`

可视化设置：

- 深度/GT 列统一使用 `Spectral_r`；
- 三列 inverse-depth 图共用同一 robust scale：
  - `vmin=0.012500000186264515`
  - `vmax=0.1680505260825158`
- GT 点半径：`5 px`，不叠 RGB 背景。

10 张预览 aggregate：

```json
{
  "num_samples": 10,
  "official_mean_absrel": 0.1251730921845124,
  "official_mean_d1": 0.8522318804946067,
  "teacher_mean_absrel": 0.11548499247944813,
  "teacher_mean_d1": 0.863819778435483
}
```

初步观察：

- 在这 10 张暗/过曝样本上，teacher_rgb 分支平均 AbsRel 更低、平均 D1 更高；
- 单样本上并非全赢，例如 `2018-02-04_21-37-51_00200` 的 D1 略低，`2018-02-10_16-39-47_00200` 的 AbsRel / D1 也略差；
- 这说明 teacher_rgb 可能有帮助，但还需要更大样本统计，并且要确认 companded vs decompanded raw normalization 哪个更适合。

后续建议：

1. 人工查看 contact sheet，确认 teacher_rgb 是否过灰、过粉或过曝；
2. 增加一个 `--raw-normalization decompanded` 的 10 张对照，判断 sensor-linear 渲染是否更稳；
3. 若视觉和指标都成立，再扩展到 100-200 张暗/过曝样本做统计；
4. 最后再考虑是否全量生成 STF teacher_rgb DAv2-L pseudo depth manifest。

### 0.4 Dark-only + 暗光增强 DAv2-L 预览（Codex 更新）

更新时间：2026-06-11 02:18（Asia/Shanghai）

按新的关注点只看 dark 样本，并在 0.3 的基础上新增两列：

1. `enhanced RGB`：从 STF 官方 dark RGB 做暗光增强；
2. `DAV2-L enhanced RGB inv`：增强 RGB 输入 DAv2-L 后的 inverse-depth prediction。

增强配方：

- source：dataset official RGB；
- recipe：LAB 空间 L 通道 CLAHE + 全局白点 percentile stretch + RGB gamma brighten；
- `clahe_clip=2.0`
- `clahe_grid=8`
- `white_percentile=99.5`
- `gamma=0.65`

运行命令：

```bash
conda run -n dav3 python finetune_stf/scripts/make_stf_teacher_rgb_dav2_dark_overexp_compare.py \
  --dark-count 10 \
  --overexp-count 0 \
  --output-dir finetune_stf/analysis/0611_stf_dark_enhance_dav2_compare \
  --tile-width 320 \
  --overwrite-preds
```

输出目录：

```text
/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_dark_enhance_dav2_compare
```

主要产物：

- contact sheet：
  - `stf_teacher_rgb_enhanced_dav2_10dark_0overexp_spectral_r.png`
- metrics CSV：
  - `selected_dark_overexp_10_metrics.csv`
- summary JSON：
  - `summary.json`
- 中间结果：
  - `enhanced_rgb/*.png`
  - `enhanced_rgb_dav2l_pred/*.npy`

10 张 dark aggregate：

```json
{
  "num_samples": 10,
  "official_mean_absrel": 0.10788008325361935,
  "official_mean_d1": 0.8733744041034095,
  "teacher_mean_absrel": 0.10145322131910925,
  "teacher_mean_d1": 0.8764321446421601,
  "enhanced_mean_absrel": 0.11370881266307346,
  "enhanced_mean_d1": 0.8591592196447195
}
```

初步观察：

- 这版增强能把暗部拉亮，但会同步放大车灯、路面高亮和暗部噪声；
- 在 10 张最暗样本上，增强分支平均 **不如** 原始 dataset RGB，也不如 raw teacher_rgb；
- raw teacher_rgb 仍是三路中平均最好的一条；
- 单样本上增强有收益，例如 `2018-02-04_21-37-51_00200`、`2018-02-04_21-37-51_00300`，但不稳定。

后续建议：

1. 若继续试增强，优先调弱增强强度，例如提高 `gamma` 到 `0.8` 或降低 CLAHE；
2. 增强分支最好先做 100 张 dark-only 统计，不建议直接全量；
3. raw teacher_rgb 当前更值得作为下一步候选。

### 0.5 Dark RAW rendering sweep：去白幕与清晰度调参（Codex 更新）

更新时间：2026-06-11 02:30（Asia/Shanghai）

问题：STF raw 按 ROD `teacher_bright_degreen_v1` 直接渲染后有明显“白幕 / milky veil”。直接增强官方 RGB 不稳定，因此改为在 RAW 渲染阶段调参。

新增脚本：

- [sweep_stf_raw_teacher_renderings_dark.py](../../finetune_stf/scripts/sweep_stf_raw_teacher_renderings_dark.py)

核心思路：

- 从 STF rectified RAW `[R, Gr, Gb, B]` 投影成 `[R, 0.5*(Gr+Gb), B]`；
- 在 raw RGB 上先做 black percentile / pedestal removal，再做 white percentile stretch；
- 使用较弱 brighten gamma，避免 ROD `gamma=0.454545` 把 STF 夜间图整体抬成白幕；
- 可选 unsharp mask 轻量锐化，帮助 DAv2-L 看见边界。

#### 0.5.1 粗 sweep

输出目录：

```text
/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_raw_render_sweep_dark10
```

主要产物：

- `stf_raw_render_variants_10dark.png`
- `stf_raw_render_dav2l_depth_variants_10dark_spectral_r.png`
- `raw_render_sweep_metrics_long.csv`
- `summary.json`

配方对比 aggregate：

| branch | mean AbsRel | mean D1 | AbsRel wins |
|---|---:|---:|---:|
| `dataset_rgb` | 0.1078800833 | 0.8733744041 | 2 |
| `raw_rod_like_g045` | 0.1014532213 | 0.8764321446 | 3 |
| `raw_black01_g065` | 0.1046477852 | 0.8679006235 | 0 |
| `raw_black05_g075` | 0.1093539176 | 0.8588743203 | 0 |
| `raw_black10_g085` | 0.1287355525 | 0.8325058779 | 0 |
| `raw_black05_g075_unsharp` | **0.0986005595** | **0.8768454861** | **5** |
| `raw_decomp_black05_g075` | 0.1090977519 | 0.8626748254 | 0 |

观察：

- 原始 `raw_rod_like_g045` 白幕最明显，但已经比 dataset RGB 平均更好；
- 单纯提高 black percentile 到 5%/10% 会压掉白幕，但过暗后 depth 指标下降；
- `raw_black05_g075_unsharp` 在视觉清晰度和 depth 指标之间最好；
- decompanded raw 这组没有带来收益，当前 companded 渲染更适合 DAv2-L 输入。

#### 0.5.2 focus sweep

在粗 sweep 最优附近继续扫：

```bash
conda run -n dav3 python finetune_stf/scripts/sweep_stf_raw_teacher_renderings_dark.py \
  --recipe-set focus \
  --output-dir finetune_stf/analysis/0611_stf_raw_render_focus_sweep_dark10 \
  --tile-width 220 \
  --overwrite-preds
```

输出目录：

```text
/home/caq/6666_raw/dav2_raw_0603/finetune_stf/analysis/0611_stf_raw_render_focus_sweep_dark10
```

focus aggregate：

| branch | mean AbsRel | mean D1 | AbsRel wins |
|---|---:|---:|---:|
| `dataset_rgb` | 0.1078800833 | 0.8733744041 | 1 |
| `raw_rod_like_g045` | 0.1014532213 | 0.8764321446 | 2 |
| `raw_b03_g065_us06` | 0.1001018426 | **0.8778080676** | 1 |
| `raw_b03_g070_us06` | 0.1002160119 | 0.8774594842 | 0 |
| `raw_b03_g075_us06` | 0.1000391144 | 0.8775817792 | 0 |
| `raw_b05_g070_us06` | 0.1007410036 | 0.8712151348 | 0 |
| `raw_b05_g075_us06` | 0.1000737620 | 0.8730334016 | 0 |
| `raw_b05_g075_us085` | **0.0986005595** | 0.8768454861 | **4** |
| `raw_b05_g080_us06` | 0.0997829596 | 0.8743575656 | 1 |
| `raw_b08_g080_us06` | 0.1110091689 | 0.8555436791 | 1 |

当前推荐候选：

```text
raw_b05_g075_us085
```

对应配方：

- raw normalization：`companded`
- black percentile：`5.0`
- white percentile：`99.8`
- gamma：`0.75`
- gains：`[1.08, 0.95, 1.10]`
- unsharp sigma：`1.1`
- unsharp amount：`0.85`

结论：

- STF 的“白幕”主要来自 ROD-like 低 gamma 强提亮叠加 STF 夜间 raw pedestal / 雾化背景；先去 black pedestal，再用较弱 gamma 能明显压白幕；
- 去得太狠会导致场景过暗，DAV2-L 结构反而变差；
- 当前最稳的折中是 `black=5% + gamma=0.75 + unsharp=0.85`；
- 如果更偏向 D1，可试 `raw_b03_g065_us06`，但 mean AbsRel 不如推荐候选。

后续建议：

1. 用 `raw_b05_g075_us085` 在 100 张 dark 样本上跑统计；
2. 若 100 张仍稳定，再生成 dark subset 的 teacher_rgb DAv2-L pseudo manifest；
3. 暂不建议全量直接替换，白天/过曝样本可能不需要同一套 black percentile。

## 1. 目标

针对 Seeing Through Fog（STF）数据集，生成可用于后续训练的 dense pseudo depth label。

STF 提供 sparse LiDAR / depth GT，但空间覆盖很稀疏，不能直接作为 dense depth supervision。当前版本不再使用 PromptDA，而是改用 **Prior Depth Anything / PriorDA**。

核心目标是：

> 使用 STF 官方 RGB 图像作为视觉输入，使用 STF sparse LiDAR depth 作为 metric prior，使用 DAv2-L relative inverse depth 作为 dense geometric prior，并通过 Prior Depth Anything 融合二者，生成 dense metric pseudo depth label。

最终希望得到的 label 具有：

1. **metric correctness**：由 STF sparse LiDAR 提供真实尺度锚点；
2. **dense structure**：由 DAv2-L 提供完整的场景结构；
3. **clear object boundary**：尽量保留 monocular depth foundation model 的物体边界和语义结构；
4. **PriorDA refinement**：由 PriorDA 对 sparse metric prior 和 dense geometric prediction 进行融合与细化；
5. **输出单位为 meters**；
6. **不把结果称为真正 dense GT**，而称为 dense pseudo label 或 LiDAR-anchored dense pseudo depth。

---

## 2. 为什么从 PromptDA 换到 Prior Depth Anything

### 2.1 你的任务本质

你现在的问题可以抽象为：

```text
STF sparse LiDAR metric depth:
    准确，有真实尺度，但非常稀疏

DAv2-L relative inverse depth:
    稠密，结构完整，边界清楚，但没有 metric scale

目标:
    融合 sparse metric depth 和 dense relative depth，
    得到 dense metric pseudo depth
```

Prior Depth Anything 的方法设定正好对应这个问题：它的核心是结合 **incomplete but precise metric information** 和 **relative but complete geometric structures**，生成 dense、detailed、metric depth map。

因此，相比 PromptDA，PriorDA 更适合作为主方案。

### 2.2 PromptDA 和 PriorDA 的区别

| 维度 | PromptDA | Prior Depth Anything |
|---|---|---|
| 主要输入形式 | RGB + low-res LiDAR prompt | RGB + metric prior + geometric depth prediction |
| 更适合的问题 | 低分辨率 depth prompt 引导 metric depth | 稀疏/不完整 metric prior 与 dense relative prediction 融合 |
| 对 DAv2-L prior 的利用 | 需要人为构造 low-res prompt | 可以直接作为 geometric input |
| 对 sparse LiDAR 的利用 | 需要设计 prompt filling 规则 | 可作为原生 metric prior |
| 和 STF 当前任务的匹配度 | 可用，但不够自然 | 更自然，更贴合 |
| 第一版建议 | 作为保底方案 | 作为主方案 |

### 2.3 使用外部 DAv2-L geometric 的真实动机

PriorDA 自带的 frozen MDE 默认是 `vitb`，conditioned / fine 网络默认也是 `vitb`。当前公开代码里 `conditioned_model_size=vitl/vitg` 仍会直接报 `coming soon`，因此第一版不能假设 PriorDA fine stage 有 ViT-L 级别的细节上限。

这也是外部提供 `geometric` 的主要收益：

1. 当传入 `geometric` 时，PriorDA 在 `DepthCompletion.preprocess()` 里跳过内部 frozen ViT-B MDE 的 forward，直接执行 `depth2disparity(geometric_depths)`；
2. DAv2-L 的 dense structure 替代 PriorDA 内置 DAv2-B / ViT-B 结构预测；
3. PriorDA 仍负责把 sparse metric prior 与 dense geometric structure 融合、对齐和 refine；
4. 但最终 fine/refinement 网络仍是 ViT-B，不能把结果理解成完整的 ViT-L metric depth。

注意：这不是说 frozen MDE checkpoint 完全不需要。当前 PriorDA `PriorDepthAnything.__init__()` 无论是否传入 `geometric`，都会初始化 `DepthCompletion` 并下载/加载 `depth_anything_v2_vitb.pth`。`geometric` 只跳过推理时的 frozen MDE 前向计算，不跳过初始化加载。

因此本方案的关键不是“手工把 DAv2-L 拟合成完美 metric depth”，而是把 DAv2-L 的相对几何结构稳定地传给 PriorDA，让 PriorDA 内部完成 metric alignment。

---

## 3. 本版本边界

### 3.1 本版本做什么

本版本只负责 dense pseudo label generation：

1. 读取 STF 官方 RGB；
2. 读取 STF sparse LiDAR depth；
3. 复用已经生成的 DAv2-L relative inverse depth；
4. 将 DAv2-L relative inverse depth 转成 positive geometric depth proxy；
5. 将 DAv2-L geometric proxy 作为 PriorDA 的 `geometric` 输入；
6. 将 STF sparse LiDAR depth 作为 PriorDA 的 `prior` 输入；
7. 使用 PriorDA 生成 dense metric depth；
8. 保存 PriorDA raw output 和可选 LiDAR overwrite / consistency-gated overwrite 版本；
9. 保存 dense label、geometric proxy、sparse mask、geometric meta、PriorDA meta、QA meta 和 debug visualization。

### 3.2 本版本暂时不做什么

暂时不做：

1. 不再使用 PromptDA；
2. 不加入 Marigold-DC、CompletionFormer、NLSPN、GuideNet 等其他模型；
3. 不做多模型大规模比较；
4. 不设计后续 training loss；
5. 不生成 continuous confidence map；
6. 不把 PriorDA 输出称为 dense GT。

推荐命名：

- dense pseudo depth label
- LiDAR-anchored dense pseudo depth
- PriorDA-generated dense pseudo supervision
- sparse-GT-calibrated dense metric depth

不推荐命名：

- dense GT
- dense ground truth
- true dense label

### 3.3 环境与安装前置条件

按项目规范默认使用 conda 环境：

```bash
conda activate dav3
```

当前本机探测结论：

1. 当前 shell 在 `base`，`base` 里 `scipy` 与 `numpy 2.x` 有版本警告，不应作为 PriorDA 环境；
2. `dav3` 存在，且 `torch` / CUDA 可用；
3. `dav3` 里目前缺少 `prior_depth_anything` 和 `torch_cluster`；
4. PriorDA 依赖 `torch_cluster` 做 KNN，对 `torch` / CUDA 版本敏感，必须在目标环境里单独验证。

安装原则：

1. 不使用 `base`；
2. 主环境优先使用 `dav3`；
3. 先确认 `dav3` 的 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.is_available()`；
4. `torch_cluster` 必须安装与当前 `torch` / CUDA 匹配的 wheel，或在该环境内编译；
5. 不要盲目让 PriorDA 的 `setup.py` 自动降级/替换现有 `torch`，除非明确决定新建独立 PriorDA 环境；
6. 若 `dav3` 无法安装兼容 `torch_cluster`，应停下来确认是否新建专用 `priorda` 环境，不要静默切环境。

源码里的 `requirements.txt` / `setup.py` 固定了 `torch==2.2.2`、`torchvision==0.17.2`、`numpy==1.25.2`、`torch_cluster==1.6.3`。因此：

1. 在 `dav3` 里不能直接 `pip install -r requirements.txt`；
2. 在 `dav3` 里不能直接 `pip install -e .`，除非确认会使用 `--no-deps`；
3. 如果当前 `torch 2.10.0+cu128` 找不到可用的 `torch_cluster` wheel，应优先新建专用 PriorDA 环境并显式记录该环境，而不是破坏 `dav3`。

代码仓库和权重仓库必须分开记录：

```text
PriorDA code repo:    SpatialVision/Prior-Depth-Anything
PriorDA weights repo: Rain729/Prior-Depth-Anything
```

第一版推荐：

```bash
cd /home/caq/6666_raw/dav2_raw_0603
git clone https://github.com/SpatialVision/Prior-Depth-Anything codex_tmp/Prior-Depth-Anything
cd codex_tmp/Prior-Depth-Anything

conda activate dav3
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 先按 dav3 的 torch/CUDA 版本安装匹配的 torch_cluster。
# 具体 wheel URL 必须与上一步输出匹配；安装后验证：
python -c "import torch_cluster; print(torch_cluster.__version__)"

# PriorDA 本体建议在确认依赖后安装，避免自动改写 torch。
pip install -e . --no-deps
python -c "from prior_depth_anything import PriorDepthAnything; print('ok')"
```

权重下载优先交给 PriorDA 的 `hf_hub_download` 自动缓存；正式跑之前要显式记录缓存到的文件名：

```text
depth_anything_v2_vitb.pth
prior_depth_anything_vitb_1_1.pth
```

安装完成后的最小验证：

1. 跑通 PriorDA 官方 sample；
2. 确认 `infer_one_sample()` 返回全分辨率 depth；
3. 确认输出单位随输入 sparse metric prior 保持 meters；
4. 确认 `.npy` 输入是普通 H x W `float32` 数组，而不是 `.npz` 容器。

---

## 4. 输入数据选择

### 4.1 RGB 输入

建议固定使用：

```text
cam_stereo_left_lut
```

理由：

1. 这是 STF 官方提供的 left-view RGB / LUT 图像；
2. 它与 left-view sparse LiDAR projection 对齐；
3. 适合作为 DAv2-L 和 PriorDA 的 RGB 输入；
4. label generation 阶段不应使用你自己的 RAW ISP 结果，否则 dense label 会和后续研究的 ISP 变量耦合；
5. RAW 只作为后续模型输入，不参与 pseudo label generation。

### 4.2 Sparse depth 输入

第一版建议使用：

```text
lidar_hdl64_last_stereo_left
```

备用或后续对照：

```text
lidar_hdl64_strongest_stereo_left
```

第一版建议只跑 `last_stereo_left`，不要引入 last / strongest fusion 规则。  
如果后续发现雾天或反射物体上 `last` 不稳定，再检查 `strongest` 或构造双 echo 过滤策略。

---

## 5. PriorDA 输入定义

PriorDA 这一版建议使用两个 depth 输入：

```text
prior     = STF sparse LiDAR metric depth
geometric = DAv2-L geometric depth proxy
```

### 5.1 `prior`

`prior` 是真实 sparse metric prior：

```text
source_sparse_path = lidar_hdl64_last_stereo_left/*.npz
prior_path        = sparse_lidar_prior/*.npy
```

建议格式：

```text
shape: H × W
dtype: float32
unit: meters
invalid value: 0
```

读取源 STF sparse depth 时必须明确：

```python
with np.load(source_sparse_path, allow_pickle=False) as data:
    Z_sparse_raw = data["arr_0"].astype(np.float32)
```

用于 PriorDA 的 `prior` 要先做显式 range filtering：

```python
M_all = np.isfinite(Z_sparse_raw) & (Z_sparse_raw >= Z_min) & (Z_sparse_raw <= Z_max)
prior = np.zeros_like(Z_sparse_raw, dtype=np.float32)
prior[M_all] = Z_sparse_raw[M_all]
```

注意：

1. valid pixel 使用真实 LiDAR depth；
2. invalid pixel 建议填 0；
3. 同时保存一个独立的 valid mask，不能只依赖 0 值推断；
4. sparse depth 不要做 bilinear interpolation；
5. 如果需要 resize，sparse map 只能用 nearest 或重新投影，不能用普通插值产生假点；
6. `> Z_max` 的 LiDAR 点不要传给 PriorDA prior，也不要参与 hold-out QA，除非该实验显式把 `Z_max` 改成 100m 并记录。

### 5.2 `geometric`

`geometric` 是由 DAv2-L relative inverse depth 构造的 dense geometric depth proxy：

```text
source_dav2_path = pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/*.npy
geometric_path   = dav2_geometric_proxy/*.npy
```

建议格式：

```text
shape: H × W
dtype: float32
unit: proxy depth, not final metric label
valid range after proxy mapping: [Z_min, Z_max]
```

现有 DAv2-L `.npy` 文件是普通 numpy array，不是 `.npz` 容器：

```python
r = np.load(source_dav2_path, allow_pickle=False).astype(np.float32)
```

不要写成 `np.load(source_dav2_path)["arr_0"]`。`["arr_0"]` 只适用于 STF sparse LiDAR `.npz`。

这个输入的作用是给 PriorDA 提供完整的几何结构和物体边界。metric scale 由 PriorDA 内部利用 sparse prior 做 global / KNN alignment，不在这里手工 robust fit。

---

## 6. DAv2-L 复用与 geometric proxy 构造

### 6.1 复用现有 DAv2-L 产物，不默认重跑

项目里已经存在全分辨率 DAv2-L RGB-LUT 产物：

```text
/mnt/drive/3333_raw/seeing_through_fog/
  pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/
    stf_rgb_lut_manifest_6216.csv
```

关键性质：

1. 样本数：`6216`；
2. 分辨率：`1024 × 1920`；
3. 存储：每个 `pseudo_depth_npy` 是普通 `.npy` array；
4. 数值语义：DAv2-L relative inverse depth / disparity-like，非 meters，数值动态范围约为几百；
5. 已被项目中其它 STF 训练和评估流程复用。

第一版应直接复用这批 `r`，保证 geometric prior 与项目其它环节使用同一份 DAv2-L 输出。

若后续坚持重跑 DAv2-L，必须显式 pin：

```text
encoder: vitl
checkpoint: /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vitl.pth
input_size: 必须写入 run_config，不允许靠脚本默认
target_resolution_policy: original image resolution, 1024 x 1920
source RGB: cam_stereo_left_lut
```

重跑输出必须单独命名，不能覆盖既有 `pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417`。

### 6.2 为什么不再做外部 robust metric fitting

PriorDA 的源码逻辑是：

```text
if geometric is provided:
    pred_disparities = depth2disparity(geometric_depths)
else:
    pred_disparities = internal frozen MDE(image)

then:
    global / KNN alignment uses sparse metric prior internally
```

因此外部 `geometric` 只需要提供 dense geometry。metric scale / shift 由 PriorDA 内部根据 sparse LiDAR prior 对齐。

本计划不再做：

```text
1 / Z = a * r + b
```

作为 PriorDA 前处理。那会重复 PriorDA 的核心 coarse alignment，还会把一套额外的 robust fitting 超参数引入 label generation。

保留的检查只有：

1. `r` 是否 finite；
2. `r` 的 polarity 是否仍是 inverse depth；
3. `r` 与 sparse inverse depth 的 rank correlation；
4. geometric proxy 是否 full resolution；
5. geometric proxy 是否数值有限、无大面积饱和。

### 6.3 Geometric proxy 构造

设：

```text
r = 现有 DAv2-L relative inverse depth
Z_min = 1.0
Z_max = 80.0
```

推荐构造一个单调的 positive depth proxy，使 PriorDA 内部 `depth2disparity(geometric)` 后仍保留 DAv2-L 的 inverse-depth 结构：

```python
r = np.load(dav2_npy_path, allow_pickle=False).astype(np.float32)
finite_pos = np.isfinite(r) & (r > 0)
if finite_pos.sum() == 0:
    raise ValueError(f"{dav2_npy_path} has no finite positive DAv2 values")

r_lo = np.percentile(r[finite_pos], 0.1)
r_hi = np.percentile(r[finite_pos], 99.9)
if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
    raise ValueError(f"Invalid DAv2 percentile range: r_lo={r_lo}, r_hi={r_hi}")

# 非有限 / 非正像素（如 sky 的 r≈0、NaN、inf）统一按“最远”处理。
# 必须在 clip 前替换，否则 np.clip(NaN, ...) 会把 NaN 传播进 proxy。
r_filled = np.where(finite_pos, r, r_lo)
r_clip = np.clip(r_filled, r_lo, r_hi)
r_unit = (r_clip - r_lo) / max(r_hi - r_lo, 1e-6)

# 若 polarity 检查显示 r 越大越远，则显式翻转；正常 DAv2-L inverse depth 不翻转。
# r_unit = 1.0 - r_unit

disp_min = 1.0 / Z_max
disp_max = 1.0 / Z_min
disp_proxy = disp_min + r_unit * (disp_max - disp_min)
Z_geo_proxy = 1.0 / disp_proxy
# 构造上 disp_proxy ∈ [disp_min, disp_max]，Z_geo_proxy 已落在 [Z_min, Z_max]；
# 下面这步 clip 只是安全网，不应承担主要的范围控制职责。
Z_geo_proxy = np.clip(Z_geo_proxy, Z_min, Z_max).astype(np.float32)
```

注意：

1. 这里的 `Z_geo_proxy` 不是 metric label；
2. `[Z_min, Z_max]` 是给 PriorDA 输入的数值范围约束，不代表 DAv2-L 已经被外部 metric calibrated；
3. 不要直接对裸 `1 / r` 做 `[1,80]` 裁剪：`r` 不是 metric 量，实测前 5 个 STF manifest 样本时，这种写法会把约 `84%~95%` 像素压到 `1m`、约 `4.6%~15.9%` 像素压到 `80m`，几乎抹掉所有结构；必须先做 6.3 这种单调 range mapping；
4. 非有限 / 非正像素的处理策略（这里按最远）会改变 sky 区域的 proxy 取值，属于 label-generation 语义，要在 `geometric_meta.json` 记录，不要靠隐式默认；
5. `r_lo/r_hi`、polarity、饱和比例必须写入 `geometric_meta.json`。

> 已在真实样本上做过 sanity：前 5 个 STF manifest 样本中，把上述 `Z_geo_proxy` 反演成 disparity 后做 PriorDA 式 global least-squares 对齐，在 LiDAR 有效点上的 AbsRel 与“直接对齐裸 `r`”只差约 `1e-5~6e-5`（例如首样本 `0.099117` vs `0.099145`），且对齐后无负 disparity。这说明 percentile proxy 基本不损失用于 PriorDA 对齐的单调几何信息；metric scale 仍由 PriorDA 内部 alignment 恢复。

### 6.4 深度范围策略

```text
Z_min = 1.0 m
Z_max = 80.0 m
```

第一版所有和 metric depth 有关的 mask 都使用同一套范围：

```python
M_all = np.isfinite(Z_sparse_raw) & (Z_sparse_raw >= Z_min) & (Z_sparse_raw <= Z_max)
```

具体策略：

1. sparse prior：只保留 `[Z_min, Z_max]`，其它位置写 0；
2. sparse valid mask：只标记 `[Z_min, Z_max]` 内的点；
3. hold-out QA：只从 `[Z_min, Z_max]` 内的点划分；
4. overwrite：默认只允许覆盖 `[Z_min, Z_max]` 内的点；
5. geometric proxy：通过 6.3 的单调映射落到 `[Z_min, Z_max]`。

如果后续发现 STF 中 80m 之外的结构对训练非常重要，可以额外尝试：

```text
Z_max = 100.0 m
```

但这是新的实验语义参数，必须显式写入正式 launch/config，不能靠路径名或代码默认值切换。

### 6.5 Prior 清洗与 failure policy

robust 逻辑从“外部 DAv2 metric fitting”移动到 sparse prior 清洗和 QA：

建议记录：

```text
num_sparse_raw_points
num_sparse_in_range_points
num_sparse_removed_by_range
spearman_corr_r_invz_on_sparse
geometric_proxy_saturation_min_ratio
geometric_proxy_saturation_max_ratio
```

如果出现以下情况，建议该样本标记为失败或低可信：

1. `[Z_min, Z_max]` 内 valid sparse points 数量太少；
2. DAv2 `r` 大面积非 finite 或非正；
3. `spearman_corr(r[M_all], 1 / Z_sparse_raw[M_all])` 明显为负；
4. geometric proxy 大面积饱和在 `Z_min` 或 `Z_max`；
5. sparse prior 与 PriorDA raw output 的 hold-out 误差异常；
6. debug visualization 显示动态物体、反射或雾区存在明显坏点。

建议最低阈值：

```text
num_valid_sparse_points >= 100
num_holdout_eval_points >= 50
```

如果 STF 某些样本 sparse point 远少于这个阈值，可以降低阈值，但必须在 `qa_meta.json` 中记录。

---

## 7. PriorDA 调用方式

### 7.1 推荐版本

建议优先使用 PriorDA v1.1 checkpoint。  
官方 README 说明 v1.1 将原来的 error map condition 替换为 sparse mask condition，并报告其在 dense patterns 上表现更好。

### 7.2 推荐调用形式

以 Python API 为准，CLI 只作为 smoke / 手工验证入口。Python API 可以直接传 numpy array，能避免 `.npz` / `.npy` loader 细节差异。

推荐逻辑形式如下：

```python
from prior_depth_anything import PriorDepthAnything

priorda = PriorDepthAnything(
    device="cuda:0",
    version="1.1",
    frozen_model_size="vitb",
    conditioned_model_size="vitb",
)

output = priorda.infer_one_sample(
    image=rgb_uint8_or_path,
    prior=sparse_lidar_prior_float32,      # H x W, meters, invalid=0
    geometric=dav2_geometric_proxy_float32, # H x W, positive depth proxy
    pattern=None,
    visualize=False,
)

Z_priorda_raw = output.detach().float().cpu().numpy().astype(np.float32)
assert Z_priorda_raw.shape == sparse_lidar_prior_float32.shape
if not np.isfinite(Z_priorda_raw).all() or (Z_priorda_raw <= 0).any():
    # 不要静默修；先记录 invalid ratio，并把样本标为 qa_failed / low_quality。
    pass
```

`pattern=None` 表示 PriorDA 使用 `prior` 中所有 valid points 作为 sparse condition。正式生成 label 时使用全部 in-range LiDAR 点；QA hold-out 时只把 train split 的 LiDAR 点写入 `prior`，把 eval split 留出来评估。

RGB 颜色通道顺序要小心：PriorDA 的 sampler 对 `str` 路径用 `PIL.Image.open`（RGB），对传入的 `np.ndarray` 则直接按给定顺序使用。本项目既有 DAv2 脚本用 `cv2.imread`（BGR）。因此 `image` 要么直接传 `cam_stereo_left_lut` 的 PNG 路径（让 PriorDA 用 PIL 读成 RGB），要么传 array 时显式转成 RGB，不能把 cv2 的 BGR array 直接喂进去。供给了 `geometric` 时内部 frozen MDE forward 会被跳过，但 fine/conditioned 网络仍吃这张 RGB，通道顺序错了会影响 refine 质量。

源码级注意：

1. `infer_one_sample()` 返回 `torch.Tensor`，不是 numpy array；
2. 当前实现返回 `pred_depth.squeeze()`，单样本正常是 `H × W`；
3. `SparseSampler` 对 `.npy` prior / geometric 直接 `np.load()`，不会自动 `astype(np.float32)`，所以写入这些 `.npy` 时必须保证 dtype；
4. `PriorDepthAnything.__init__()` 仍会加载 frozen MDE checkpoint，即使之后 `geometric` 分支跳过 frozen MDE forward；
5. CLI 的 `--coarse_only` 用 `type=bool`，容易把字符串 `"0"` 解析成 `True`，正式批量不建议依赖 CLI。

如果使用 CLI，逻辑形式如下：

```bash
priorda test \
  --image_path path/to/cam_stereo_left_lut.png \
  --prior_path path/to/sparse_lidar_prior.npy \
  --geometric_path path/to/dav2_geometric_proxy.npy \
  --visualize 1
```

注意：具体命令参数以当前安装版本的 PriorDA repo 为准。  
正式批量跑之前，应先用 3 到 5 个 STF 样本确认：

1. sparse LiDAR 源 `.npz` 是否通过 `["arr_0"]` 正确读取；
2. PriorDA 输入 `.npy` 是否是普通 H x W `float32` array；
3. invalid value = 0 是否被正确忽略；
4. `geometric` 输入是否跳过内部 frozen MDE forward 并按预期生效；
5. 输出 tensor 是否 `.detach().cpu().numpy()` 后保存；
6. 输出分辨率是否为 `1024 × 1920`，与 RGB 一致；
7. 输出 depth 单位是否保持 meters。

---

## 8. 最终 dense label 构造

PriorDA 输出记为：

```text
Z_priorda_raw
```

建议至少保存这些版本：

```text
dense_depth_priorda_raw.npy
dense_depth_priorda_lidar_overwrite.npy
dense_depth_priorda_lidar_gated_overwrite.npy  # optional
```

### 8.1 No-overwrite PriorDA output

```text
Z_priorda_raw = PriorDA(RGB, sparse_lidar_prior, dav2_geometric_proxy)
```

这是 PriorDA 未做 LiDAR overwrite 的输出。为了和第一版 `[Z_min, Z_max]` 监督范围一致，保存为 `dense_depth_priorda_raw.npy` 前建议做 range clip：

```python
Z_raw_model = Z_priorda_raw.copy()
Z_raw = np.clip(Z_raw_model, Z_min, Z_max).astype(np.float32)
```

`Z_raw_model` 的 min/max、非有限/非正比例要写入 `priorda_meta.json`。如果后续确实需要完全未裁剪的模型输出，再新增 `priorda_model_output_unclipped/`，不要复用 `dense_depth_priorda_raw/` 这个训练标签目录。

### 8.2 LiDAR-overwrite final label

为了保证真实 sparse LiDAR GT 点不被网络输出覆盖，可以构造 overwrite 版本：

```text
Z_final = Z_priorda_raw
Z_final[sparse_valid_mask] = sparse_lidar_depth[sparse_valid_mask]
```

也就是：

```text
valid sparse LiDAR pixels:
    使用 STF sparse LiDAR depth

其他 pixels:
    使用 PriorDA dense prediction
```

这样最终 label 同时具有：

1. sparse GT 点上的物理测量准确性；
2. 非 sparse 区域的 dense prediction；
3. 清晰的物体结构和完整场景覆盖。

但 LiDAR overwrite 有一个明确风险：稀疏 LiDAR 点可能在动态物体、反射区域、遮挡边界或雾中回波处是坏点；无条件 overwrite 还会在 dense prediction 上制造 salt-and-pepper 深度突变。

因此第一版必须同时保留：

```text
dense_depth_priorda_raw.npy
dense_depth_priorda_lidar_overwrite.npy
```

如果 QA 发现突变明显，推荐增加 consistency-gated overwrite：

```text
overwrite_mask =
    sparse_valid_mask
    & isfinite(Z_priorda_raw)
    & (abs(Z_priorda_raw - Z_sparse_raw) / Z_sparse_raw <= overwrite_rel_threshold)
```

第一版可先记录 gated mask 和 unconditional overwrite 的差异；训练时优先使用哪一版，应在下游实验 config 中显式写出，不能靠目录名隐式决定。

---

## 9. 是否生成 confidence map

第一版不建议生成 continuous confidence map。

原因：

1. PriorDA 的标准输出主要是 depth，不是 depth + uncertainty；
2. DAv2-L relative inverse depth 本身没有可靠 uncertainty；
3. sparse LiDAR mask 可以保存，但不能直接等价于 dense confidence；
4. 手工构造 confidence map 会引入许多主观规则；
5. 当前目标是先生成稳定 dense label，不应额外增加变量。

第一版建议保存以下替代信息：

```text
sparse_lidar_mask.png
geometric_meta.json
priorda_meta.json
qa_meta.json
debug_vis.png
```

后续如果训练阶段确实需要 confidence，可以再基于以下信息构造：

1. sparse LiDAR valid mask；
2. DAv2 geometric proxy polarity / saturation；
3. PriorDA output 与 hold-out sparse LiDAR 的误差；
4. depth range；
5. weather / illumination metadata；
6. local image gradient 与 depth discontinuity 一致性；
7. sky / reflective / fog-heavy regions。

但这不属于第一版 label generation。

---

## 10. 输出文件设计

输出根目录必须带 launch timestamp，格式遵守项目规范 `MMDD_HHMM`。建议：

```text
/mnt/drive/3333_raw/seeing_through_fog/
  <MMDD_HHMM>_pseudo_depth_priorda_dav2l_geo_lidar_last_6216/
    run_config.json
    manifest.csv

    dense_depth_priorda_raw/
      sample_xxx.npy

    dense_depth_priorda_lidar_overwrite/
      sample_xxx.npy

    dense_depth_priorda_lidar_gated_overwrite/
      sample_xxx.npy

    dav2_geometric_proxy/
      sample_xxx.npy

    sparse_lidar_prior/
      sample_xxx.npy

    sparse_lidar_mask/
      sample_xxx.png

    geometric_meta/
      sample_xxx.json

    priorda_meta/
      sample_xxx.json

    qa_meta/
      sample_xxx.json

    debug_vis/
      sample_xxx.png
```

如果只是 smoke test，输出路径必须包含 `smoke` / `debug` / `tmp` / `codex_smoke` 之一；成功后只删除这些明确临时路径。

### 10.0 全量前 10 样本 preview gate

在任何 6216 全量生成之前，必须先跑 10 个 STF 样本，把 depth completion 后的结果展示出来供人工确认。这个步骤不是可选项。

推荐输出根目录：

```text
/mnt/drive/3333_raw/seeing_through_fog/
  debug_priorda_preview10_<MMDD_HHMM>/
    preview10_manifest.csv
    preview10_summary.json
    preview10_contact_sheet.png

    dense_depth_priorda_raw/
      sample_xxx.npy

    dense_depth_priorda_lidar_overwrite/
      sample_xxx.npy

    dav2_geometric_proxy/
      sample_xxx.npy

    sparse_lidar_prior/
      sample_xxx.npy

    debug_vis/
      sample_xxx.png
```

说明：

1. 目录名包含 `debug`，但这是用户审阅用 preview artifact，不是成功后自动删除的 smoke artifact；
2. `preview10_manifest.csv` 必须记录 10 个 `sample_id`、split、RGB、sparse depth、DAv2-L npy、输出 panel 路径；
3. 10 个样本要确定性选择，不能每次随机变。建议按 `sha1(sample_id)` 排序后取前 10，或手工指定并写入 manifest；
4. 预览优先展示 `dense_depth_priorda_raw`，因为它是真正的 PriorDA depth completion 结果；overwrite 版本只作为对照；
5. `preview10_contact_sheet.png` 至少包含每个样本的 RGB、sparse LiDAR overlay、DAv2 geometric proxy、PriorDA raw completion、LiDAR overwrite、raw-vs-overwrite diff；
6. 只有 preview 结果经人工确认没有明显结构崩坏、尺度崩坏、边界异常或 overwrite 点状突变后，才启动正式全量任务。

建议 10 样本 preview 命令也放 tmux，避免首次下载权重或 CUDA 编译/加载卡住前台：

```bash
TS=$(date +%m%d_%H%M)
SESSION="priorda_stf_preview10_${TS}"
OUT_ROOT="/mnt/drive/3333_raw/seeing_through_fog/debug_priorda_preview10_${TS}"
LOG_PATH="/home/caq/6666_raw/dav2_raw_0603/logs/${TS}_priorda_stf_preview10.log"
BATCH_SCRIPT="/home/caq/6666_raw/dav2_raw_0603/path/to/batch_script.py"

mkdir -p "$(dirname "${LOG_PATH}")"
tmux new -s "${SESSION}" -d "bash -lc 'conda activate dav3 && python \"${BATCH_SCRIPT}\" --max-samples 10 --preview-contact-sheet --output-root \"${OUT_ROOT}\" 2>&1 | tee \"${LOG_PATH}\"'"
```

启动后必须报告：

```text
tmux attach -t ${SESSION}
tail -f ${LOG_PATH}
preview contact sheet: ${OUT_ROOT}/preview10_contact_sheet.png
```

正式 6216 样本批量生成预计是长任务，必须按项目规范放到 tmux：

```bash
TS=$(date +%m%d_%H%M)
SESSION="priorda_stf_${TS}"
OUT_ROOT="/mnt/drive/3333_raw/seeing_through_fog/${TS}_pseudo_depth_priorda_dav2l_geo_lidar_last_6216"
LOG_PATH="/home/caq/6666_raw/dav2_raw_0603/logs/${TS}_priorda_stf_generate.log"
BATCH_SCRIPT="/home/caq/6666_raw/dav2_raw_0603/path/to/batch_script.py"

mkdir -p "$(dirname "${LOG_PATH}")"
tmux new -s "${SESSION}" -d "bash -lc 'conda activate dav3 && python \"${BATCH_SCRIPT}\" --output-root \"${OUT_ROOT}\" 2>&1 | tee \"${LOG_PATH}\"'"
```

启动后必须报告：

```text
tmux attach -t ${SESSION}
tail -f ${LOG_PATH}
```

### 10.1 dense_depth_priorda_raw

PriorDA 未做 LiDAR overwrite 的 dense label 版本；保存前按 `[Z_min, Z_max]` clip，避免训练标签范围与 sparse prior/QA 范围不一致。

```text
format: .npy
dtype: float32
unit: meters
shape: H × W
```

用途：

1. debug；
2. 对比 LiDAR overwrite 前后差异；
3. 检查 PriorDA 是否在 hold-out sparse GT 点附近偏离过大；
4. 后续 ablation；
5. 避免 unconditional overwrite 的 salt-and-pepper 风险。

注意：这里的 `raw` 表示 no-overwrite，不表示完全未裁剪的模型 tensor。

### 10.2 dense_depth_priorda_lidar_overwrite

LiDAR 无条件 overwrite 版本。

```text
Z_final = Z_priorda_raw
Z_final[sparse_valid_mask] = sparse_lidar_depth[sparse_valid_mask]
```

用途：作为一个保守版本保存，但不默认等价于训练首选。训练时是否使用它必须在实验 config 中显式写出。

### 10.3 dense_depth_priorda_lidar_gated_overwrite

可选的 consistency-gated overwrite 版本。

用途：

1. 只在 sparse LiDAR 与 PriorDA raw output 局部一致时覆盖；
2. 降低动态物体 / 反射坏点导致的突变；
3. 与 unconditional overwrite 做 ablation。

### 10.4 dav2_geometric_proxy

DAv2-L relative inverse depth 经单调 proxy mapping 后得到的 dense positive depth proxy。

用途：

1. 作为 PriorDA 的 `geometric` 输入；
2. 检查 DAv2-L dense structure 和物体边界；
3. 和 PriorDA output 做对比；
4. 分析 PriorDA 是否过度平滑或过度 hallucinate。

### 10.5 sparse_lidar_prior

STF sparse LiDAR metric prior。

```text
valid pixels: in-range LiDAR depth in meters
invalid pixels: 0
```

读取源数据：

```python
with np.load(sparse_depth_path, allow_pickle=False) as data:
    Z_sparse_raw = data["arr_0"].astype(np.float32)
```

保存给 PriorDA 的 prior 应是普通 `.npy` array：

```python
np.save(prior_out_path, prior.astype(np.float32))
```

### 10.6 sparse_lidar_mask

```text
format: .png
dtype: uint8
0: invalid
255: valid
```

用途：

1. 标记真实 sparse GT 点；
2. 后续区分 measured depth 和 pseudo depth；
3. 做 LiDAR overwrite；
4. 做 hold-out sparse-point QA。

### 10.7 geometric_meta

建议内容：

```json
{
  "sample_id": "xxx",
  "rgb_path": "...",
  "sparse_depth_path": "...",
  "dav2_source_manifest": "/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv",
  "dav2_source_npy": "...",
  "dav2_model": "Depth Anything V2 Large relative inverse depth",
  "dav2_reused_existing_output": true,
  "geometric_transform": "percentile_normalized_inverse_proxy",
  "external_metric_alignment": false,
  "metric_alignment_owner": "PriorDA internal global/KNN alignment",
  "z_min": 1.0,
  "z_max": 80.0,
  "r_percentile_low": 0.0,
  "r_percentile_high": 0.0,
  "r_nonfinite_or_nonpositive_ratio": 0.0,
  "r_nonfinite_policy": "fill_with_r_percentile_low_before_clipping",
  "polarity_flipped": false,
  "num_sparse_raw_points": 0,
  "num_sparse_in_range_points": 0,
  "spearman_corr_r_invz": 0.0,
  "geometric_proxy_min": 1.0,
  "geometric_proxy_max": 80.0,
  "geometric_proxy_saturation_min_ratio": 0.0,
  "geometric_proxy_saturation_max_ratio": 0.0,
  "notes": ""
}
```

### 10.8 priorda_meta

建议内容：

```json
{
  "sample_id": "xxx",
  "priorda_repo": "SpatialVision/Prior-Depth-Anything",
  "priorda_repo_commit": "8c029cbca669443fe0bbf8dcefb5f91ad531084d or installed commit",
  "priorda_weights_repo": "Rain729/Prior-Depth-Anything",
  "priorda_checkpoint": "prior_depth_anything_vitb_1_1.pth",
  "frozen_mde_checkpoint": "depth_anything_v2_vitb.pth",
  "frozen_mde_loaded_at_init": true,
  "frozen_mde_forward_skipped_by_geometric": true,
  "priorda_version": "1.1",
  "frozen_model_size": "vitb",
  "conditioned_model_size": "vitb",
  "torch_version": "",
  "torch_cuda_version": "",
  "torch_cluster_version": "",
  "input_prior": "sparse_lidar_prior",
  "input_geometric": "dav2_geometric_proxy",
  "pattern": null,
  "output_raw": "dense_depth_priorda_raw",
  "output_raw_meaning": "no_lidar_overwrite_post_clipped",
  "output_final": "dense_depth_priorda_lidar_overwrite",
  "lidar_overwrite": true,
  "gated_overwrite": false,
  "output_unit": "meters",
  "output_shape": [0, 0],
  "output_nonfinite_or_nonpositive_ratio": 0.0,
  "output_raw_min": 0.0,
  "output_raw_max": 0.0,
  "post_clip_min": 1.0,
  "post_clip_max": 80.0,
  "notes": ""
}
```

---

## 11. Debug visualization

建议每个样本保存一张 debug 图：

```text
debug_vis/sample_xxx.png
```

建议包含：

1. RGB；
2. sparse LiDAR overlay；
3. DAv2-L raw relative inverse depth；
4. DAv2 geometric depth proxy；
5. PriorDA raw output；
6. unconditional LiDAR overwrite output；
7. gated overwrite output 或 gated mask；
8. hold-out LiDAR eval overlay；
9. sparse valid mask。

最关键要看：

1. 物体边界是否被过度平滑；
2. 车辆、行人、路牌、树、路沿是否保留清楚；
3. hold-out sparse LiDAR 点附近 PriorDA 是否偏离过大；
4. 雾天远距离区域是否出现不合理结构；
5. 夜间高亮 / 反射区域是否出现严重错误；
6. sky 或无效区域是否出现异常近距离深度；
7. unconditional overwrite 是否造成明显 salt-and-pepper 点状突变。

---

## 12. 推荐第一版配置表

| 项目 | 第一版配置 |
|---|---|
| Dataset | Seeing Through Fog |
| RGB 输入 | `cam_stereo_left_lut` |
| Sparse prior | `lidar_hdl64_last_stereo_left` |
| Dense geometric prior | 复用既有 DAv2-L relative inverse depth |
| DAv2-L source | `pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv` |
| Geometric transform | percentile-normalized inverse proxy |
| External metric fitting | 不做 |
| Metric alignment | PriorDA 内部 global / KNN alignment |
| Depth range | `[1m, 80m]` |
| PriorDA input `prior` | sparse LiDAR metric depth, invalid=0 |
| PriorDA input `geometric` | DAv2 geometric depth proxy |
| PriorDA code repo | `SpatialVision/Prior-Depth-Anything` |
| PriorDA weights repo | `Rain729/Prior-Depth-Anything` |
| PriorDA checkpoint | `prior_depth_anything_vitb_1_1.pth` |
| Frozen / fine model size | `vitb` / `vitb` |
| Fine-stage 分辨率 | `PRIORDA_FINE_HEIT=native`（hires-fine，去糊）；原默认 518 会模糊 |
| Pattern | `None` for production |
| Final output | clipped no-overwrite + unconditional overwrite + optional gated overwrite |
| Full-run gate | 先跑 `debug_priorda_preview10_<MMDD_HHMM>` 并展示 `preview10_contact_sheet.png` |
| Output unit | meters |
| Output format | `.npy` float32 |
| Confidence map | 第一版不做 |
| QA | hold-out sparse LiDAR eval |
| Debug output | mask + geometric meta + PriorDA meta + QA meta + visualization |

---

## 13. 推荐 pipeline

```python
for row in manifest_rows:
    sample_id = row["sample_name"]
    rgb_path = row["rgb_path"]  # 推荐传路径，让 PriorDA 用 PIL 读 RGB。

    with np.load(row["sparse_depth_path"], allow_pickle=False) as data:
        Z_sparse_raw = data["arr_0"].astype(np.float32)

    M_all = np.isfinite(Z_sparse_raw) & (Z_sparse_raw >= Z_min) & (Z_sparse_raw <= Z_max)
    prior = np.zeros_like(Z_sparse_raw, dtype=np.float32)
    prior[M_all] = Z_sparse_raw[M_all]

    # Full-resolution DAv2-L relative inverse depth, not meters.
    r = np.load(row["pseudo_depth_npy"], allow_pickle=False).astype(np.float32)
    geometric = build_dav2_geometric_proxy(r, Z_min, Z_max).astype(np.float32)

    np.save(sparse_lidar_prior_path(sample_id), prior.astype(np.float32))
    np.save(dav2_geometric_proxy_path(sample_id), geometric.astype(np.float32))

    output = priorda.infer_one_sample(
        image=rgb_path,
        prior=prior,
        geometric=geometric,
        pattern=None,
        visualize=False,
    )
    Z_priorda_raw = output.detach().float().cpu().numpy().astype(np.float32)
    assert Z_priorda_raw.shape == Z_sparse_raw.shape

    invalid_output = ~np.isfinite(Z_priorda_raw) | (Z_priorda_raw <= 0)
    if invalid_output.any():
        # 不要静默修；记录 invalid ratio，并把样本标为 qa_failed / low_quality。
        pass

    Z_raw_model = Z_priorda_raw.copy()
    Z_raw = np.clip(Z_raw_model, Z_min, Z_max).astype(np.float32)
    Z_overwrite = Z_raw.copy()
    Z_overwrite[M_all] = Z_sparse_raw[M_all]
    Z_gated = optional_consistency_gated_overwrite(Z_raw, Z_sparse_raw, M_all)

    np.save(dense_depth_priorda_raw_path(sample_id), Z_raw.astype(np.float32))
    np.save(dense_depth_priorda_lidar_overwrite_path(sample_id), Z_overwrite.astype(np.float32))
    np.save(dense_depth_priorda_lidar_gated_overwrite_path(sample_id), Z_gated.astype(np.float32))

    sparse_mask_png = (M_all.astype(np.uint8) * 255)
    save_png(sparse_lidar_mask_path(sample_id), sparse_mask_png)
    write_json(geometric_meta_path(sample_id), geometric_meta)
    write_json(priorda_meta_path(sample_id), priorda_meta)
    write_json(qa_meta_path(sample_id), qa_meta)
    save_debug_vis(debug_vis_path(sample_id), debug_panel)
```

---

## 14. 最低限度 QA，不作为完整实验流程

你之前明确说暂时不做 Phase 1 / Phase 2，这里只建议保留非常轻量的 per-sample QA，用于发现坏样本，不作为正式实验。

不能只在传给 PriorDA 的 sparse 点上评估，因为 `pattern=None` 时 PriorDA 会使用全部 prior 点做 condition，并且 coarse KNN map 会在这些点 cover；如果后面又执行 LiDAR overwrite，`Z_final[M] = Z_sparse[M]` 更会让指标变成自循环。

第一版建议增加 hold-out sparse LiDAR QA：

```python
import hashlib

coords = np.argwhere(M_all)
if len(coords) < 100:
    raise ValueError(f"{sample_id}: too few in-range sparse points: {len(coords)}")
seed = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
rng = np.random.default_rng(seed)
perm = rng.permutation(len(coords))
min_prior_points = 5  # PriorDA default K; use priorda.args.K if accessible.
num_holdout = max(50, int(round(len(coords) * 0.2)))
num_holdout = min(num_holdout, len(coords) - min_prior_points)
holdout_coords = coords[perm[:num_holdout]]

M_holdout = np.zeros_like(M_all, dtype=bool)
M_holdout[holdout_coords[:, 0], holdout_coords[:, 1]] = True
M_prior = M_all & ~M_holdout

prior_for_qa = np.zeros_like(Z_sparse_raw, dtype=np.float32)
prior_for_qa[M_prior] = Z_sparse_raw[M_prior]

output_qa = priorda.infer_one_sample(
    image=rgb_path,
    prior=prior_for_qa,
    geometric=geometric,
    pattern=None,
    visualize=False,
)
Z_priorda_raw_qa = output_qa.detach().float().cpu().numpy().astype(np.float32)
```

在 `M_holdout` 上记录：

```text
eval_mask = M_holdout & np.isfinite(Z_priorda_raw_qa) & (Z_priorda_raw_qa > 0)
AbsRel_holdout = mean(abs(Z_priorda_raw_qa[eval_mask] - Z_sparse_raw[eval_mask]) / Z_sparse_raw[eval_mask])
RMSE_holdout   = sqrt(mean((Z_priorda_raw_qa[eval_mask] - Z_sparse_raw[eval_mask])^2))
delta1_holdout = mean(max(Z_priorda_raw_qa[eval_mask] / Z_sparse_raw[eval_mask],
                         Z_sparse_raw[eval_mask] / Z_priorda_raw_qa[eval_mask]) < 1.25)
```

正式 label 生成仍可用全部 `M_all` 作为 prior；hold-out QA 只用于发现坏样本和调试，不作为最终 label。

建议在 `qa_meta.json` 里记录：

```json
{
  "qa_mode": "holdout_sparse_lidar",
  "holdout_fraction": 0.2,
  "holdout_seed_policy": "sha1(sample_id) first 8 hex chars -> uint32 seed",
  "num_prior_points": 0,
  "num_holdout_points": 0,
  "absrel_holdout": 0.0,
  "rmse_holdout": 0.0,
  "delta1_holdout": 0.0,
  "absrel_raw_on_all_sparse_for_reference_only": 0.0,
  "notes": ""
}
```

`absrel_raw_on_all_sparse_for_reference_only` 只能作为参考，不应作为主要质量指标。

---

## 15. 关键风险与处理

### 15.1 DAv2 polarity 风险

虽然当前设定认为 DAv2-L 输出是 relative inverse depth，但不同代码接口可能返回不同 polarity。  
建议在少量样本上检查：

```text
corr(r[M_all], 1/Z_sparse_raw[M_all])
```

如果为正，说明 `r` 越大越近，符合 inverse depth。  
如果为负，说明需要翻转或重新定义 `r`。

### 15.2 PriorDA 输入格式风险

PriorDA 示例支持 `.png` 和 `.npy` 输入，但不同版本 loader 对 invalid depth 的处理可能有细节差异。  
批量跑之前必须确认：

1. STF 源 sparse `.npz` 是否用 `["arr_0"]` 正确读取；
2. PriorDA 输入 prior / geometric 是否保存为普通 `.npy` H x W array；
3. invalid=0 是否不会被当成真实 0m；
4. `geometric` 是否真的生效，并跳过内部 frozen MDE forward；
5. 输出是否与 RGB 分辨率一致；
6. 输出单位是否与输入 prior 一致。

### 15.3 Sparse LiDAR 投影错误

STF sparse LiDAR 点可能在动态物体边界、反射区域、透明区域、雾中回波区域存在不稳定。  
建议至少做：

1. depth range filtering；
2. hold-out QA；
3. optional consistency-gated overwrite；
4. debug visualization；
5. 对异常样本标记 `low_quality` 或 `qa_failed`。

### 15.4 LiDAR overwrite 不连续风险

无条件 LiDAR overwrite 会把单像素 sparse measurement 硬贴到 dense prediction 上，可能制造局部深度突变。尤其在以下区域需要谨慎：

1. 车辆 / 行人动态边界；
2. 玻璃、反光牌、湿地面；
3. 重雾远距离回波；
4. RGB-LiDAR 投影轻微错位处。

因此必须保存 raw output；如果 overwrite 版本用于训练，要在实验 config 中显式写 `label_variant=dense_depth_priorda_lidar_overwrite` 或 `label_variant=dense_depth_priorda_lidar_gated_overwrite`。

### 15.5 环境与依赖风险

PriorDA 依赖 `torch_cluster`，安装失败或与 torch/CUDA ABI 不匹配会在 KNN 阶段失败。落地前必须在 `dav3` 里完成：

1. `import prior_depth_anything`；
2. `import torch_cluster`；
3. 官方 sample inference；
4. 3-5 个 STF smoke sample inference；
5. 10 个 STF preview depth completion，并生成 `preview10_contact_sheet.png` 给人工确认。

### 15.6 RGB teacher bias

PriorDA 和 DAv2 都依赖 RGB 图像，因此最终 pseudo label 会带有 RGB teacher bias。  
后续如果用它监督 RAW 模型，应避免声称它是真实 dense GT。  
最终评价仍应优先基于真实 sparse LiDAR points 或其他真实 metric measurements。

---

## 16. 和 PromptDA 版本相比的主要改动

| 项目 | PromptDA 旧方案 | PriorDA 新方案 |
|---|---|---|
| 主模型 | PromptDA | Prior Depth Anything |
| sparse LiDAR 用法 | 构造 low-res prompt，并用 LiDAR median overwrite prompt cell | 直接作为 PriorDA `prior` |
| DAv2 用法 | 先 alignment，再下采样成 prompt | 复用既有 DAv2-L，构造 geometric proxy 后作为 `geometric` input |
| prompt 尺寸 | 需要设计 192×256 prompt | 不再需要 low-res prompt |
| metric 对齐 | 外部构造 prompt 规则 | PriorDA 内部 global / KNN alignment |
| confidence map | 不做 | 不做 |
| final label | PromptDA output | PriorDA raw + sparse LiDAR overwrite / gated overwrite |
| 解释逻辑 | 人为构造 prompt | metric prior + dense geometric prediction 融合，更自然 |

补充说明：PriorDA 的 `geometric` 只需提供 dense geometric structure；对 DAv2-L relative inverse depth 做反演和单调 range mapping 即可。metric 对齐由 PriorDA 内部完成，不要照旧写一套外部 robust affine fitting。

---

## 17. 一句话总结

PriorDA 版本的核心方案是：

> 将 STF sparse LiDAR depth 作为 metric prior，复用既有 DAv2-L relative inverse depth 并构造 dense geometric proxy，使用 Prior Depth Anything 内部 alignment / refinement 融合二者生成 dense metric pseudo depth，同时保存 raw、LiDAR overwrite 和可选 gated overwrite 版本。

这个方案比 PromptDA 更适合当前 STF 场景，因为它更直接地建模了：

```text
sparse but metric LiDAR
+
dense but relative monocular depth
→
dense metric pseudo depth
```

---

## 18. 参考来源

1. Prior Depth Anything GitHub repository: https://github.com/SpatialVision/Prior-Depth-Anything
2. Prior Depth Anything project page: https://prior-depth-anything.github.io/
3. Paper: Depth Anything with Any Prior: https://arxiv.org/abs/2505.10565
4. Prior Depth Anything Hugging Face weights: https://huggingface.co/Rain729/Prior-Depth-Anything
5. Depth Anything V2 GitHub repository: https://github.com/DepthAnything/Depth-Anything-V2

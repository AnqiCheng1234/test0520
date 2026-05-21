# DAV2-S Pseudo Label 替换计划

## Summary
目标是生成一套新的 STF DAV2-S pseudo depth label，并用它替换当前 `pseudo_depth_dav2_official_vitl...` manifest 做后续训练对照。默认在本机 `/home/caq/6666_raw/dav2_raw_0520` 执行，使用 conda 环境 `dav3`，不覆盖旧 DAV2-L label，不处理 STF 全部 12997 张图，只复刻旧 manifest 的 6216 个样本。

## Key Changes
- 新增生成脚本：`scripts/generate_stf_dav2_pseudo.py`
  - 模板基必须拆成两部分：DAV2 推理逻辑取自 `scripts/generate_lod_day_dav2_pseudo.py`，包括 `MODEL_CONFIGS`、`build_model()`、`DepthAnythingV2.infer_image()` 调用和 `--encoder vits|vitb|vitl|vitg` choices；STF manifest I/O、metadata、`atomic_save_*`、`run_config.json`、`run_summary.json`、progress 结构取自 `scripts/generate_stf_da3_pseudo.py`。
  - 不直接 fork 成 DA3 接口残留版：不能保留 DA3 专用的 `model.inference(images, process_res=..., ref_view_strategy=..., use_ray_pose=...)` 调用，也不能复制 DA3 的 depth units 字面值。
  - 输入旧 manifest，输出新的 `.npy/.png/stf_rgb_lut_manifest_6216.csv/stf_rgb_lut_inputs_6216.txt/run_config.json/run_summary.json`。
  - DAV2 配置固定支持 `--encoder vits|vitb|vitl|vitg`，本次用 `--encoder vits`。
  - 保存 `DepthAnythingV2.infer_image()` 的原始 float32 输出，分辨率保持 RGB-LUT 原图 `(1024, 1920)`。
  - 默认 `--input-size 518`，对齐 DAV2 official 推理习惯；可视化仅用于检查，训练只读 `.npy`。
  - `DEPTH_VALUE_UNITS` 必须重写为 DAV2 相对反深度/disparity 语义，例如 `"value": "relative_inverse_depth_from_dav2"`、`"direction": "larger_is_closer"`；不能复制 DA3 的 `"affine_invariant_depth_from_da3mono"` / `"larger_is_farther"`。
  - metadata 写清楚 checkpoint path/hash、encoder、input_size、depth units 为 DAV2 inverse-relative/disparity。
- 新增训练队列脚本：`finetune_stf/scripts/0521_run_stf_lora_full_dav2s_queue.sh`
  - 从现有 `0521_run_stf_lora_full_da3_queue.sh` 派生。
  - 默认 `PRETRAINED=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`。这里的 `PRETRAINED` 是 student backbone 初始化，和 pseudo teacher 解耦；本次 student 仍是 `vits`，`pretrained_from` 实际不变，只把 teacher pseudo label 从 DAV2-L manifest 换成 DAV2-S manifest。
  - 新变量 `DAV2S_MANIFEST` 指向新生成 manifest。
  - `require_file` 列表改为只检查 `"${PRETRAINED}"` 和 `"${DAV2S_MANIFEST}"`；去掉 DA3 manifest，避免新队列被不相关的 DA3 文件卡住。
  - 只跑 exp1-6 的 DAV2 pseudo 对照矩阵，run suffix 全部改成 `pseudovits`，避免继续写成 `pseudovitl`。
  - 不复跑 exp7-8 的 DA3 pseudo 对照，因为本次只替换 DAV2 teacher label，不影响 DA3 两行。
  - smoke 改为 DAV2-S target 的两个代表路径：exp1 `rgb_lora decoder` 和 exp3 `raw_ram_rgb_lora decoder`。虽然原 DA3 queue smoke 只覆盖 exp7-8，但新 DAV2-S manifest 会影响 exp1-6，需要重新 smoke。

## Operational Steps
1. 预检查：
   ```bash
   cd /home/caq/6666_raw/dav2_raw_0520
   conda env list | grep dav3
   test -f /home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
   test -f /mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv
   avail_g=$(df -BG --output=avail /mnt/drive | tail -1 | tr -dc '0-9')
   [ "${avail_g:-0}" -ge 70 ] || { echo "Need >=70G free on /mnt/drive, got ${avail_g}G"; exit 1; }
   ```
   要求 `/mnt/drive` 至少预留 `70G`，因为新 DAV2-S label 预计接近旧目录 `48G`，还要留出 manifest/png/log 等余量。

2. 实现 `scripts/generate_stf_dav2_pseudo.py` 后，先跑生成 smoke：
   ```bash
   SRC=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv
   CKPT=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth
   SMOKE_OUT=/mnt/drive/3333_raw/seeing_through_fog/codex_smoke_pseudo_depth_dav2_vits_rgb_lut_6216_$(date +%m%d_%H%M)

   conda run --live-stream -n dav3 python scripts/generate_stf_dav2_pseudo.py \
     --manifest "$SRC" \
     --output-root "$SMOKE_OUT" \
     --checkpoint "$CKPT" \
     --encoder vits \
     --input-size 518 \
     --device cuda \
     --max-samples 8

   python - <<PY
   import csv, numpy as np, pathlib
   out = pathlib.Path("$SMOKE_OUT")
   rows = list(csv.DictReader(open(out / "stf_rgb_lut_manifest_6216.csv")))
   assert len(rows) == 8, len(rows)
   for row in (rows[0], rows[-1]):
       p = row["pseudo_depth_npy"]
       assert p.startswith(str(out)), p
       assert "pseudo_depth_dav2_official_vitl" not in p, p
       a = np.load(p)
       assert a.shape == (1024, 1920), a.shape
       assert a.dtype == np.float32, a.dtype
       assert np.isfinite(a).all(), p
   print("smoke ok", len(rows), rows[0]["pseudo_depth_npy"], rows[-1]["pseudo_depth_npy"])
   PY
   ```
   断言通过后删除 `$SMOKE_OUT`。失败则保留该目录排查。

3. 正式生成必须放 tmux：
   ```bash
   TS=$(date +%m%d_%H%M)
   OUT=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vits_rgb_lut_6216_${TS}
   SESSION=${TS}_gen_stf_dav2s_pseudo
   LOG=/home/caq/6666_raw/dav2_raw_0520/logs/${SESSION}.log
   SRC=/mnt/drive/3333_raw/seeing_through_fog/pseudo_depth_dav2_official_vitl_rgb_lut_6216_20260417/stf_rgb_lut_manifest_6216.csv
   CKPT=/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth

   mkdir -p logs
   tmux new-session -d -s "$SESSION" \
     "cd /home/caq/6666_raw/dav2_raw_0520 && conda run --live-stream -n dav3 python scripts/generate_stf_dav2_pseudo.py --manifest '$SRC' --output-root '$OUT' --checkpoint '$CKPT' --encoder vits --input-size 518 --device cuda 2>&1 | tee -a '$LOG'"
   ```
   `LOG=/home/caq/6666_raw/dav2_raw_0520/logs/${SESSION}.log` 与现有零散日志放在 `logs/` 下的惯例一致。
   报告/记录：
   ```bash
   tmux attach -t "$SESSION"
   tail -f "$LOG"
   ```

4. 正式生成完成后验收：
   ```bash
   test -f "$OUT/stf_rgb_lut_manifest_6216.csv"
   test "$(find "$OUT" -maxdepth 1 -name '*.npy' | wc -l)" -eq 6216
   test "$(find "$OUT" -maxdepth 1 -name '*.png' | wc -l)" -eq 6216
   python - <<PY
   import csv, json, numpy as np, pathlib
   out = pathlib.Path("$OUT")
   summary = json.load(open(out / "run_summary.json"))
   assert summary["status"] == "completed", summary
   assert summary["generated"] == 6216, summary
   assert summary["failed"] == 0, summary
   assert summary["source_sample_count"] == 6216, summary
   assert summary["sample_count"] == 6216, summary
   rows = list(csv.DictReader(open(out / "stf_rgb_lut_manifest_6216.csv")))
   assert len(rows) == 6216, len(rows)
   for row in (rows[0], rows[-1]):
       p = row["pseudo_depth_npy"]
       assert p.startswith(str(out)), p
       assert "pseudo_depth_dav2_official_vitl" not in p, p
       a = np.load(p)
       print(p, a.shape, a.dtype, np.isfinite(a).all(), float(a.min()), float(a.max()))
       assert a.shape == (1024, 1920), a.shape
       assert a.dtype == np.float32, a.dtype
       assert np.isfinite(a).all(), p
   print("summary ok", summary["status"], summary["generated"])
   PY
   ```

5. 实现 DAV2-S 训练队列脚本后，先 smoke，再跑正式 exp1-6：
   ```bash
   DAV2S_MANIFEST="$OUT/stf_rgb_lut_manifest_6216.csv" \
   SESSION_PREFIX=stf_0521_dav2s_pseudo_seq \
   RUN_SMOKES=1 \
   START_FORMAL_EXP=1 \
   END_FORMAL_EXP=6 \
   bash finetune_stf/scripts/0521_run_stf_lora_full_dav2s_queue.sh
   ```
   该脚本启动后记录它打印的 tmux session、queue log、attach、tail 命令。正式 run 名必须由脚本用启动时刻 `MMDD_HHMM` 生成，并包含 `pseudovits`。

## Test Plan
- 生成脚本 smoke：`--max-samples 8`，成功后删除 `codex_smoke...` 输出目录。
- 生成结果验收：`run_summary.json` 断言 `status=completed/generated=6216/failed=0/source_sample_count=6216/sample_count=6216`；manifest 行数 `6216`；`.npy/.png` 各 `6216`；首尾 `.npy` 路径都在新 `$OUT` 下且不再指向旧 `vitl` 目录；首尾 `.npy` 均为 `(1024,1920) float32` 且全 finite。
- 训练 smoke：两个代表训练路径，输出路径必须包含 `codex_smoke`，成功后只删除 smoke artifacts。
- 正式训练验收：每个 `pseudovits` run 的 `config.json` 中 `stf_pseudo_manifest` 指向新 DAV2-S manifest，`encoder=vits`，`pretrained_from` 仍指向 DAV2-S student backbone checkpoint `/home/caq/333_cvpr/da_ours/checkpoints/depth_anything_v2_vits.pth`。

## Assumptions
- 默认本机执行；如果切到 186，需要先同步新增脚本和 DAV2-S checkpoint，再用 186 的本地路径重跑同一流程。
- 旧 DAV2-L pseudo label 保留，用于 `pseudovitl` 对照，不修改 `DEFAULT_STF_PSEUDO_MANIFEST`。
- 本次只改变 pseudo teacher label，从 DAV2-L 换成 DAV2-S；训练 backbone/student 仍保持 `encoder=vits` 和现有超参矩阵。

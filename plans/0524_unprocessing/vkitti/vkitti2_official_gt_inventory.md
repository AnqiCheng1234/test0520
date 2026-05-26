# VKITTI2 official GT inventory and local subset note

Checked on 2026-05-24.

This note records the official VKITTI2 ground-truth inventory versus the current local subset under:

```text
/mnt/drive/1111_new_works/VKITTI2
```

## 1. Official VKITTI2 contents

Official source:

```text
https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
https://europe.naverlabs.com/wp-content/uploads/2020/01/vkitti2.pdf
```

The official VKITTI2 dataset provides one archive per data type:

```text
vkitti_2.0.3_rgb.tar
vkitti_2.0.3_depth.tar
vkitti_2.0.3_classSegmentation.tar
vkitti_2.0.3_instanceSegmentation.tar
vkitti_2.0.3_textgt.tar.gz
vkitti_2.0.3_forwardFlow.tar
vkitti_2.0.3_backwardFlow.tar
vkitti_2.0.3_forwardSceneFlow.tar
vkitti_2.0.3_backwardSceneFlow.tar
```

Official data types therefore include:

```text
RGB
metric depth
class segmentation
instance segmentation
forward optical flow
backward optical flow
forward scene flow
backward scene flow
text GT metadata:
  colors.txt
  extrinsic.txt
  intrinsic.txt
  info.txt
  bbox.txt
  pose.txt
```

The official format description lists paths like:

```text
SceneX/Y/frames/rgb/Camera_Z/rgb_%05d.jpg
SceneX/Y/frames/depth/Camera_Z/depth_%05d.png
SceneX/Y/frames/classsegmentation/Camera_Z/classgt_%05d.png
SceneX/Y/frames/instancesegmentation/Camera_Z/instancegt_%05d.png
SceneX/Y/frames/backwardFlow/Camera_Z/backwardFlow_%05d.png
SceneX/Y/frames/backwardSceneFlow/Camera_Z/backwardSceneFlow_%05d.png
SceneX/Y/frames/forwardFlow/Camera_Z/flow_%05d.png
SceneX/Y/frames/forwardSceneFlow/Camera_Z/sceneFlow_%05d.png
SceneX/Y/colors.txt
SceneX/Y/extrinsic.txt
SceneX/Y/intrinsic.txt
SceneX/Y/info.txt
SceneX/Y/bbox.txt
SceneX/Y/pose.txt
```

Where:

```text
X in {01, 02, 06, 18, 20}
Y in {15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right,
      clone, fog, morning, overcast, rain, sunset}
Z in {0, 1}
```

## 2. Edge GT status

VKITTI2 does not appear to provide a separate official edge archive.

If edge supervision is needed later, it should be treated as a derived target, for example:

```text
1. segmentation boundary from classgt / instancegt
2. depth discontinuity edge from depth
3. combined semantic + geometric edge
```

This derived-edge choice changes the training/evaluation semantics and should be made explicit in any formal config or launch script.

## 3. Current local subset

Current local top-level entries:

```text
rgb/
depth/
cache_raw_sensor_linear_dual_644x1008_k1rand_fp32_seed20260516/
pseudoraw_cache/
vkitti2_0417test/
```

Current local original-data counts:

```text
rgb:   42520 files
depth: 42520 files

rgb Camera_0:   21260
rgb Camera_1:   21260
depth Camera_0: 21260
depth Camera_1: 21260
```

No local files or directories were found for:

```text
classSegmentation / classsegmentation
instanceSegmentation / instancesegmentation
semantic / label
edge
forwardFlow / backwardFlow
forwardSceneFlow / backwardSceneFlow
bbox / pose / intrinsic / extrinsic / colors.txt / info.txt
```

The current project split remains:

```text
finetune_stf/dataset/splits/vkitti2/train.txt
entries: 19559
camera: Camera_0 only
fields: RGB path + depth path
```

## 4. Why the local copy is incomplete

The current repository download script only downloads RGB and depth by default:

```text
scripts/download_vkitti2.sh
```

It has URLs for:

```text
vkitti_2.0.3_rgb.tar
vkitti_2.0.3_depth.tar
vkitti_2.0.3_textgt.tar.gz
```

But `textgt` is only downloaded when:

```bash
INCLUDE_TEXTGT=1
```

The current script does not download class segmentation, instance segmentation, optical flow, or scene flow.

## 5. Official archive availability check

The following official archive URLs returned HTTP 200 during a lightweight HEAD check:

```text
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_instanceSegmentation.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_backwardFlow.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_forwardSceneFlow.tar
https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_backwardSceneFlow.tar
```

Approximate content lengths reported by the server:

```text
rgb:                    7.5 GB
depth:                  8.1 GB
classSegmentation:      1.0 GB
instanceSegmentation:   0.17 GB
textgt:                 0.025 GB
forwardFlow:            31.4 GB
backwardFlow:           29.2 GB
forwardSceneFlow:       16.0 GB
backwardSceneFlow:      16.0 GB
```

If downloading later, use tmux for flow / scene-flow archives and write logs to a clear path. These are large downloads and should not run in the foreground.

## 6. Follow-up implications

If adding official GT locally later, keep experiment-semantic choices explicit:

```text
use_class_segmentation
use_instance_segmentation
use_textgt_metadata
use_flow
use_scene_flow
derived_edge_source = none | depth | classseg | instanceseg | combined
```

Do not infer the available GT from directory names in formal experiments. Validate the chosen GT types centrally in the resolved config stage.

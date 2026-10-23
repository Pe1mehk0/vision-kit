# Vision Challenge kit

My own code for Saturday. Contains **no XRlabs data**. Never commit their data, labels or weights here.

## 0. Setup (every new session)

```python
!git clone https://github.com/<me>/vision-kit.git      # or upload kit.py
import sys; sys.path.insert(0, 'vision-kit'); from kit import *
check_env()                                           # must say CUDA True
```
If `tracking` is planned: `!pip -q install lap` now, not at 15:00.

## 1. Data (10:30)

Describe the format to Claude (folders, extensions, JSON keys, value ranges; no images, no label contents).

| Their format | Call |
|---|---|
| COCO JSON | `names = coco_to_yolo('ann.json', 'labels/')` |
| Pascal VOC XML | `names = voc_to_yolo('xml/', 'labels/')` |
| Masks (pixel = class) | `mask_to_yolo('masks/', 'labels/', {1: 0, 2: 1})` |
| Already YOLO | nothing |

Then:
```python
round_trip_test('valid/images', 'valid/labels', names)   # converters still correct (~0 difference)
label_check('data.yaml')                                  # counts per class + label_check.png: boxes on tools?
```
**From videos? Split by video before anything else:**
```python
imgs = [...]                                              # all image Paths
split_by_group(imgs, group_of=lambda p: <video id from name>, out_dir='vsplit',
               val_groups={...}, test_groups={...}, names=names)
```

## 2. Train in the background (terminal), analyse in the notebook

```bash
cd /content && nohup sh -c '
yolo train model=yolo11n.pt data=DATA.yaml epochs=50 imgsz=640 batch=16 patience=15 seed=0 project=RUNS name=base &&
yolo train model=yolo11n.pt data=DATA.yaml epochs=50 imgsz=640 batch=16 patience=15 seed=0 hsv_h=0.05 hsv_s=0.9 mixup=0.1 erasing=0.5 project=RUNS name=aug &&
yolo train model=yolo11s.pt data=DATA.yaml epochs=50 imgsz=640 batch=16 patience=15 seed=0 project=RUNS name=small
' > train.log 2>&1 &
```
Queue another run behind job PID: `nohup sh -c 'while kill -0 <PID> 2>/dev/null; do sleep 30; done; yolo train ...' > q.log 2>&1 &`

Progress: `runs_summary('RUNS')` · GPU: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`

## 3. Judge

```python
make_harsh('test/images', 'test/labels', 'harsh', names)        # check harsh/preview.png first
harsh_eval({'base': 'RUNS/base/weights/best.pt', 'aug': 'RUNS/aug/weights/best.pt'}, 'data.yaml', 'harsh')
conf_sweep('RUNS/aug/weights/best.pt', 'data.yaml')             # pick threshold from their FP/FN scoring
```
Need stronger robustness than the built-in knobs? `offline_augment('train/images', 'train/labels', 'aug/train')`, then train on it.

## 4. Speed and submit (by 16:30)

```python
ms_per_frame('RUNS/aug/weights/best.pt', 'valid/images')        # only when nothing is training
eng = export_trt('RUNS/aug/weights/best.pt', 'data.yaml')        # ~4 min first time; check mAP held
ms_per_frame(eng, 'valid/images')
```

## Rules I learned (22 Oct)

- Leaky (random) split said 0.959; split by video said 0.852 (val) and 0.632 (test). Decide only on an honest split.
- Model knobs (bigger, newer, more epochs) tied on easy data; augmentation was the big lever there. Their data decides.
- Clean ties hide robustness gaps: check `harsh_eval` before choosing between tied models.
- Per-class thresholds and tracking: keep only if they beat the simple version on held-out data.
- Never measure speed while training. After out-of-memory: `nvidia-smi`, then Runtime → Restart session (not delete).
- Don't pass a Python list of paths to `predict()` (one giant batch). Pass the folder with `stream=True`.
- Build TensorRT engines on the scoring machine. Keep `best.pt` as the master copy.
- NDA: no screenshots, no data out of their environment, delete their-data weights after; confirm deletion within 7 days.

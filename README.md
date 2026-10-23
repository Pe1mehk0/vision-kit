# Vision Challenge kit

My own code for Saturday. Contains **no XRlabs data**. Never commit their data, labels or weights here.

## 0. Setup (every new session)

**Launch the Brev instance at ~10:05** (takes ~7 min to boot). L4, 1 GPU, "VM Mode w/ Jupyter", an option with stop/start. Open JupyterLab → Notebook → Python 3.

**Cell 1 (raw Brev VM: no pip, no libGL; fixes found 23 Oct):**
```python
import sys
!{sys.executable} -m ensurepip --upgrade
!{sys.executable} -m pip install -q ultralytics lap roboflow
!sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 libglib2.0-0
!git clone https://github.com/Pe1mehk0/vision-kit.git
```
Wait until the cell counter is a number (not `[*]`), then **Kernel → Restart Kernel**.

**Cell 2:**
```python
import sys; sys.path.insert(0, 'vision-kit'); from kit import *
check_env()                                           # must say CUDA True | GPU NVIDIA L4
```
- No sudo? `!{sys.executable} -m pip uninstall -y opencv-python` then `!{sys.executable} -m pip install -q opencv-python-headless`, restart kernel.
- Colab instead of Brev: just `!pip -q install ultralytics lap` + clone.
- Paste check: an `IndentationError` right after pasting = leading spaces. Select all, Shift+Tab.

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

Jupyter: "+" tab → Other → **Terminal** (shell commands go here, not in a notebook cell).
On Brev, start with `source ~/.venv/bin/activate` so the terminal sees the installed packages.

```bash
source ~/.venv/bin/activate && cd ~ && nohup sh -c '
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
report('RUNS/aug/weights/best.pt', 'data.yaml', split='test')   # plain-text numbers, safe to paste to Claude
t = threshold_for_scoring('RUNS/aug/weights/best.pt', 'data.yaml', split='val', fp_cost=1, fn_cost=1)
#   ^ set fp_cost / fn_cost from THEIR scoring rule; tune on val, confirm on test
```
Need stronger robustness than the built-in knobs?
```python
offline_augment('train/images', 'train/labels', 'aug/train', copies=1, strength=1.0, max_ops=2)
# 10 damage types (colour casts, noise, motion, blur, dark/bright, smoke, glare, jpeg); pass ops=(...) to hold some out
```
Then train on a data.yaml whose train points at `aug/train/images`.

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
- Always check the split sizes: a downloaded dataset had 1 validation image (23 Oct). `label_check` shows it.
- Check what a box covers (tool tip vs whole tool). Don't mix datasets with different conventions.
- Jupyter counters: `[*]` running, `[5]` done, `[ ]` not run / interrupted. Don't restart the kernel while `[*]`.
- Brev billing: storage keeps charging while stopped. Delete the instance when finished.

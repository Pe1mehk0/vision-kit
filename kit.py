"""
kit.py - Real-Time Vision Challenge toolkit (Aleks, Oct 2026).
Usage in a notebook:  import sys; sys.path.insert(0, '<repo folder>'); from kit import *
Every function prints what it did. Nothing here uploads data anywhere.
"""
import json, os, shutil, collections, random, time
from pathlib import Path
import numpy as np, cv2, yaml

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

# ----------------------------------------------------------------- environment
def check_env():
    """GPU visible? Which Ultralytics version? Run this first, every session."""
    import torch, ultralytics
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - you are on CPU!'
    print(f'ultralytics {ultralytics.__version__} | torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPU {gpu}')

def _imgs(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)

def _names(data_yaml):
    n = yaml.safe_load(open(data_yaml))['names']
    return n if isinstance(n, dict) else dict(enumerate(n))

def write_data_yaml(out_yaml, root, names, train='train/images', val='valid/images', test=None):
    """Write a data.yaml with an absolute root path (avoids ../ path bugs)."""
    d = {'path': str(Path(root).absolute()), 'train': train, 'val': val}
    if test: d['test'] = test
    d['names'] = names if isinstance(names, dict) else dict(enumerate(names))
    yaml.safe_dump(d, open(out_yaml, 'w'), sort_keys=False)
    print('wrote', out_yaml); return str(out_yaml)

# ----------------------------------------------------------------- converters
def yolo_to_coco(img_dir, lbl_dir, names, out_json):
    """YOLO txt -> one COCO JSON. Used for round-trip tests."""
    images, anns, aid = [], [], 1
    names = list(names.values()) if isinstance(names, dict) else list(names)
    for iid, p in enumerate(_imgs(img_dir), start=1):
        H, W = cv2.imread(str(p)).shape[:2]
        images.append({'id': iid, 'file_name': p.name, 'width': W, 'height': H})
        lf = Path(lbl_dir) / (p.stem + '.txt')
        for line in (lf.read_text().splitlines() if lf.exists() else []):
            if not line.strip(): continue
            c, xc, yc, bw, bh = map(float, line.split()[:5]); w, h = bw * W, bh * H
            anns.append({'id': aid, 'image_id': iid, 'category_id': int(c) + 1,
                         'bbox': [xc * W - w / 2, yc * H - h / 2, w, h], 'area': w * h, 'iscrowd': 0}); aid += 1
    json.dump({'images': images, 'annotations': anns,
               'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(names)]}, open(out_json, 'w'))
    print(f'{len(images)} images, {len(anns)} boxes -> {out_json}')

def coco_to_yolo(coco_json, out_lbl_dir):
    """COCO JSON -> YOLO txt per image. Remaps category ids to 0..N-1, clips boxes, skips iscrowd / <1px.
    Returns {index: name} for data.yaml."""
    d = json.load(open(coco_json))
    idx = {cid: i for i, cid in enumerate(sorted(c['id'] for c in d['categories']))}
    names = {idx[c['id']]: c['name'] for c in d['categories']}
    imgs = {im['id']: im for im in d['images']}; lines = {i: [] for i in imgs}; skipped = 0
    for a in d['annotations']:
        if a.get('iscrowd', 0): skipped += 1; continue
        im = imgs[a['image_id']]; W, H = im['width'], im['height']; x, y, w, h = a['bbox']
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x2 - x1 < 1 or y2 - y1 < 1: skipped += 1; continue
        lines[a['image_id']].append(f"{idx[a['category_id']]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    out = Path(out_lbl_dir); out.mkdir(parents=True, exist_ok=True)
    for i, ls in lines.items(): (out / (Path(imgs[i]['file_name']).stem + '.txt')).write_text('\n'.join(ls))
    print(f'{len(imgs)} label files, {sum(map(len, lines.values()))} boxes, {skipped} skipped -> {out}')
    return names

def voc_to_yolo(xml_dir, out_lbl_dir, names=None):
    """Pascal VOC XML -> YOLO txt. names: list to fix the class order, else sorted from the files."""
    import xml.etree.ElementTree as ET
    xmls = sorted(Path(xml_dir).glob('*.xml'))
    if names is None:
        names = sorted({o.find('name').text.strip() for x in xmls for o in ET.parse(x).findall('object')})
    cid = {n: i for i, n in enumerate(names)}; out = Path(out_lbl_dir); out.mkdir(parents=True, exist_ok=True); nb = 0
    for x in xmls:
        r = ET.parse(x).getroot(); W = float(r.find('size/width').text); H = float(r.find('size/height').text); ls = []
        for o in r.findall('object'):
            b = o.find('bndbox'); x1, y1, x2, y2 = (float(b.find(t).text) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            ls.append(f"{cid[o.find('name').text.strip()]} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"); nb += 1
        (out / (x.stem + '.txt')).write_text('\n'.join(ls))
    print(f'{len(xmls)} files, {nb} boxes, classes {names}'); return dict(enumerate(names))

def mask_to_yolo(mask_dir, out_lbl_dir, class_map, min_area=50):
    """Class-index mask PNGs (pixel value = class) -> YOLO boxes, one per connected blob.
    class_map: {pixel_value: yolo_class}. Gotcha: a tool split by another object becomes 2 boxes."""
    out = Path(out_lbl_dir); out.mkdir(parents=True, exist_ok=True); total = 0; masks = sorted(Path(mask_dir).glob('*.png'))
    for mp in masks:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE); H, W = m.shape; ls = []
        for val, cls in class_map.items():
            n, _, st, _ = cv2.connectedComponentsWithStats((m == val).astype(np.uint8), connectivity=8)
            for i in range(1, n):
                x, y, w, h, area = st[i]
                if area >= min_area: ls.append(f'{cls} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}')
        (out / (mp.stem + '.txt')).write_text('\n'.join(ls)); total += len(ls)
    print(f'{len(masks)} masks -> {total} boxes in {out}')

def round_trip_test(img_dir, lbl_dir, names, tmp='/tmp/rt'):
    """YOLO -> COCO -> YOLO and compare. Largest difference should be ~0."""
    yolo_to_coco(img_dir, lbl_dir, names, f'{tmp}.json'); coco_to_yolo(f'{tmp}.json', f'{tmp}_labels')
    worst = 0; files = 0
    for lf in Path(lbl_dir).glob('*.txt'):
        a = np.loadtxt(lf, ndmin=2); b = np.loadtxt(Path(f'{tmp}_labels') / lf.name, ndmin=2)
        if a.size == 0 and b.size == 0: files += 1; continue
        assert a.shape == b.shape and (a[:, 0] == b[:, 0]).all(), f'mismatch in {lf.name}'
        worst = max(worst, np.abs(a - b).max()); files += 1
    print(f'{files} files match, largest difference {worst:.6f}')

# ----------------------------------------------------------------- data checks
def label_check(data_yaml, n_show=4, out_png='label_check.png'):
    """Counts per split and class, empty and out-of-range boxes, and n_show images with boxes drawn (saved)."""
    cfg = yaml.safe_load(open(data_yaml)); root = Path(cfg.get('path', Path(data_yaml).parent)); names = _names(data_yaml)
    shown = []
    for split in ('train', 'val', 'test'):
        if split not in cfg: continue
        img_dir = root / cfg[split]; lbl_dir = Path(str(img_dir).replace('images', 'labels'))
        imgs = _imgs(img_dir); cnt, empty, bad = collections.Counter(), 0, 0
        for p in imgs:
            lf = lbl_dir / (p.stem + '.txt')
            rows = [l.split() for l in lf.read_text().splitlines() if l.strip()] if lf.exists() else []
            if not rows: empty += 1
            for r in rows:
                cnt[names[int(r[0])]] += 1
                if any(not 0 <= float(v) <= 1 for v in r[1:5]): bad += 1
        print(f'{split}: {len(imgs)} imgs | {empty} without boxes | {bad} out-of-range boxes | ' +
              ', '.join(f'{k} {v}' for k, v in cnt.most_common()))
        if split == 'train': shown = [(p, lbl_dir) for p in random.sample(imgs, min(n_show, len(imgs)))]
    tiles = []
    for p, lbl_dir in shown:
        im = cv2.imread(str(p)); h, w = im.shape[:2]; lf = lbl_dir / (p.stem + '.txt')
        for l in (lf.read_text().splitlines() if lf.exists() else []):
            if not l.strip(): continue
            c, x, y, bw, bh = map(float, l.split()[:5])
            p1, p2 = (int((x - bw/2) * w), int((y - bh/2) * h)), (int((x + bw/2) * w), int((y + bh/2) * h))
            cv2.rectangle(im, p1, p2, (0, 255, 0), 2); cv2.putText(im, names[int(c)], (p1[0], max(12, p1[1] - 4)), 0, 0.6, (0, 255, 0), 2)
        tiles.append(cv2.resize(im, (480, int(480 * h / w))))
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        cv2.imwrite(out_png, np.hstack([cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]))
        print('drawn examples ->', out_png, '(open it: do boxes sit on the tools, with the right names?)')

def split_by_group(images, group_of, out_dir, val_groups, test_groups, names):
    """Honest split: every group (video / patient) goes to exactly one split.
    images: list of image Paths whose labels live at ../labels/<stem>.txt
    group_of: function Path -> group id (e.g. lambda p: int(p.stem[:2]))  <- decide from the file names!"""
    out = Path(out_dir); shutil.rmtree(out, ignore_errors=True)
    which = lambda g: 'valid' if g in val_groups else 'test' if g in test_groups else 'train'
    per = collections.defaultdict(collections.Counter)
    for s in ('train', 'valid', 'test'):
        for sub in ('images', 'labels'): (out / s / sub).mkdir(parents=True)
    for p in images:
        g = group_of(p); s = which(g); per[s][g] += 1
        lab = p.parent.parent / 'labels' / (p.stem + '.txt')
        shutil.copy(p, out / s / 'images' / p.name)
        if lab.exists(): shutil.copy(lab, out / s / 'labels' / lab.name)
    for s in ('train', 'valid', 'test'): print(f'{s}: {sum(per[s].values())} images from groups {sorted(per[s])}')
    return write_data_yaml(out / 'data.yaml', out, names, test='test/images')

# ----------------------------------------------------------------- runs
def runs_summary(project, names=None):
    """Best mAP50 / mAP50-95 for every run folder (reads results.csv)."""
    import pandas as pd
    for n in (names or sorted(os.listdir(project))):
        f = Path(project) / n / 'results.csv'
        if not f.exists(): print(f'{n:18s} - no results yet'); continue
        d = pd.read_csv(f); d.columns = d.columns.str.strip(); b = d.loc[d['metrics/mAP50-95(B)'].idxmax()]
        print(f"{n:18s} epoch {int(d['epoch'].iloc[-1]):3d} | best mAP50 {d['metrics/mAP50(B)'].max():.3f} | "
              f"best mAP50-95 {b['metrics/mAP50-95(B)']:.3f} (epoch {int(b['epoch'])})")

def ms_per_frame(weights, img_dir, imgsz=640, n=200):
    """Real speed: one frame at a time after 10 warm-up frames. Only on an idle GPU!"""
    from ultralytics import YOLO
    m = YOLO(weights, task='detect'); imgs = _imgs(img_dir)[:n]
    for p in imgs[:10]: m.predict(str(p), imgsz=imgsz, verbose=False)
    t = [sum(m.predict(str(p), imgsz=imgsz, verbose=False)[0].speed.values()) for p in imgs]
    print(f'{Path(weights).name}: {np.mean(t):.1f} ms/frame (median {np.median(t):.1f}) = {1000/np.mean(t):.0f} fps'); return float(np.mean(t))

def export_trt(pt, data_yaml, imgsz=640):
    """TensorRT FP16 export + accuracy check. Build on the scoring machine, never copy engines."""
    from ultralytics import YOLO
    try:    eng = YOLO(pt).export(format='engine', half=True, imgsz=imgsz, device=0)
    except TypeError: eng = YOLO(pt).export(format='engine', quantize=16, imgsz=imgsz, device=0)
    m = YOLO(eng, task='detect').val(data=data_yaml, imgsz=imgsz, batch=1, verbose=False, plots=False)
    print(f'engine {eng} | mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}  (compare with the .pt before using it)'); return eng

def conf_sweep(weights, data_yaml, split='val', grid=(0.1, 0.25, 0.4, 0.55, 0.7)):
    """Precision / recall vs confidence threshold (no retraining)."""
    from ultralytics import YOLO
    m = YOLO(weights)
    for c in grid:
        r = m.val(data=data_yaml, split=split, conf=c, verbose=False, plots=False)
        print(f'conf {c:.2f} | P {r.box.mp:.3f} | R {r.box.mr:.3f} | mAP50 {r.box.map50:.3f}')

# ----------------------------------------------------------------- robustness (sim-to-real)
_rng = np.random.default_rng(0)
def scope_mask(im):
    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 5); m = (g > 15).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = np.zeros(m.shape, np.uint8)
    if not cs: return np.ones(m.shape, np.float32)
    cv2.drawContours(out, [cv2.convexHull(max(cs, key=cv2.contourArea))], -1, 1, -1)
    return cv2.GaussianBlur(cv2.erode(out, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3)
def c_blur(im):   return cv2.GaussianBlur(im, (0, 0), 3)
def c_motion(im):
    k = np.zeros((15, 15), np.float32); k[7, :] = 1 / 15; return cv2.filter2D(im, -1, k)
def c_dark(im):   return (im.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
def c_red(im):
    f = im.astype(np.float32); f[..., 2] *= 1.25; f[..., :2] *= 0.75; return f.clip(0, 255).astype(np.uint8)
def c_noise(im):  return (im.astype(np.float32) + _rng.normal(0, 18, im.shape)).clip(0, 255).astype(np.uint8)
def c_smoke(im):
    H, W = im.shape[:2]
    f = cv2.GaussianBlur(cv2.resize(_rng.random((4, 7)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 20)
    d = ((0.25 + 0.35 * (f - f.min()) / (np.ptp(f) + 1e-6)) * scope_mask(im))[..., None]
    return (cv2.GaussianBlur(im.astype(np.float32), (0, 0), 1.5) * (1 - d) + 205 * d).clip(0, 255).astype(np.uint8)
def c_glare(im):
    H, W = im.shape[:2]; sm = scope_mask(im); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inside = g[sm > 0]; thr = np.percentile(inside, 85) if inside.size else 255
    ys, xs = np.where((sm > 0) & (g >= thr)); mask = np.zeros((H, W), np.float32)
    for i in (_rng.choice(len(xs), size=min(len(xs), int(_rng.integers(4, 9))), replace=False) if len(xs) else []):
        cv2.ellipse(mask, (int(xs[i]), int(ys[i])), (int(_rng.integers(W // 45, W // 18)), int(_rng.integers(H // 60, H // 25))),
                    float(_rng.integers(0, 180)), 0, 360, 1.0, -1)
    mask = (cv2.GaussianBlur(mask, (0, 0), 2) * sm)[..., None]
    return (im.astype(np.float32) * (1 - mask) + 250 * mask).clip(0, 255).astype(np.uint8)
CORRUPTIONS = dict(blur=c_blur, motion=c_motion, dark=c_dark, smoke=c_smoke, red=c_red, noise=c_noise, glare=c_glare)
# NOTE: smoke/glare assume a round scope view with a black border. Check the preview before trusting them.

def make_harsh(img_dir, lbl_dir, out_root, names, conds=None):
    """One damaged copy of a test set per condition (labels unchanged). Saves a preview png."""
    out_root = Path(out_root); imgs = _imgs(img_dir); conds = conds or list(CORRUPTIONS)
    for c in conds:
        d = out_root / c; (d / 'images').mkdir(parents=True, exist_ok=True)
        shutil.copytree(lbl_dir, d / 'labels', dirs_exist_ok=True)
        for p in imgs: cv2.imwrite(str(d / 'images' / p.name), CORRUPTIONS[c](cv2.imread(str(p))))
        for cache in d.glob('*.cache'): cache.unlink()
        write_data_yaml(d / 'data.yaml', d, names, train='images', val='images')
    p = imgs[0]; tiles = [cv2.resize(cv2.imread(str(p)), (320, 180))] + \
        [cv2.resize(cv2.imread(str(out_root / c / 'images' / p.name)), (320, 180)) for c in conds]
    cv2.imwrite(str(out_root / 'preview.png'), np.hstack(tiles)); print('preview ->', out_root / 'preview.png', '| order: clean,', conds)

def harsh_eval(models, clean_yaml, harsh_root, clean_split='test', conds=None):
    """mAP50 of each model on clean + every damaged set. models: {'name': 'path/to/best.pt'}"""
    from ultralytics import YOLO
    conds = conds or [c for c in CORRUPTIONS if (Path(harsh_root) / c / 'data.yaml').exists()]; rows = {}
    for mn, w in models.items():
        m = YOLO(w); rows[('clean', mn)] = m.val(data=clean_yaml, split=clean_split, verbose=False, plots=False).box.map50
        for c in conds: rows[(c, mn)] = m.val(data=str(Path(harsh_root) / c / 'data.yaml'), split='val', verbose=False, plots=False).box.map50
    print(f"{'condition':10s}" + ''.join(f'{mn:>12s}' for mn in models))
    for c in ['clean'] + conds: print(f'{c:10s}' + ''.join(f'{rows[(c, mn)]:12.3f}' for mn in models))
    print(f"{'avg damaged':10s}" + ''.join(f'{np.mean([rows[(c, mn)] for c in conds]):12.3f}' for mn in models))
    return rows

def offline_augment(train_img_dir, train_lbl_dir, out_dir, seed=1):
    """Clean + one randomly damaged copy per training image (red/noise/motion/blur/dark, random strengths)."""
    rng = np.random.default_rng(seed)
    def red(im):
        f = im.astype(np.float32); f[..., 2] *= rng.uniform(1.05, 1.3); f[..., :2] *= rng.uniform(0.7, 0.95); return f.clip(0, 255).astype(np.uint8)
    def noise(im): return (im.astype(np.float32) + rng.normal(0, rng.uniform(5, 25), im.shape)).clip(0, 255).astype(np.uint8)
    def motion(im):
        L = int(rng.integers(5, 22)); k = np.zeros((L, L), np.float32); k[L // 2, :] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D(((L-1)/2, (L-1)/2), float(rng.uniform(0, 180)), 1), (L, L)); k /= k.sum()
        return cv2.filter2D(im, -1, k)
    def blur(im): return cv2.GaussianBlur(im, (0, 0), float(rng.uniform(1, 4)))
    def dark(im): return (im.astype(np.float32) * rng.uniform(0.35, 0.8)).clip(0, 255).astype(np.uint8)
    fns = [red, noise, motion, blur, dark]; out = Path(out_dir)
    for sub in ('images', 'labels'): (out / sub).mkdir(parents=True, exist_ok=True)
    for p in _imgs(train_img_dir):
        lab = Path(train_lbl_dir) / (p.stem + '.txt'); shutil.copy(p, out / 'images' / p.name)
        if lab.exists(): shutil.copy(lab, out / 'labels' / lab.name); shutil.copy(lab, out / 'labels' / (p.stem + '_aug.txt'))
        cv2.imwrite(str(out / 'images' / (p.stem + '_aug.jpg')), fns[int(rng.integers(len(fns)))](cv2.imread(str(p))))
    print(len(_imgs(out / 'images')), 'training images in', out)

#!/usr/bin/env python3
"""
DeepCacher — train_model.py

Trains TWO models:
  1. compressibility_model.json  — binary: is this file worth compressing?
     Label: 1 if zstd achieves >1.05x ratio, 0 if not (images, encrypted, etc.)
     Used by compressor to route files: skip solid stream if incompressible.

  2. compressor_model.json       — multiclass: lz4 / zstd / brotli
     Only trained on compressible files. Used to sort files in the solid stream
     so similar-algo files are adjacent → better cross-file pattern matching.

Also trains:
  - zstd_dict.bin  — dictionary for small similar files

Usage:
    python3 train_model.py /path/to/dataset/
"""

import os, sys, math, json, argparse, warnings, time
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ── Feature extraction (same as compressor.py) ────────────────────────────────

def shannon_entropy(data):
    if not data: return 0.0
    freq = Counter(data)
    n = len(data)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def extract_features_from_bytes(data, file_size, ext):
    import numpy as np

    ent      = shannon_entropy(data)
    hist, _  = np.histogram(list(data[:65536]), bins=256, density=True)
    byte_std  = float(np.std(hist))
    byte_mean = float(np.mean(hist))
    byte_max  = float(np.max(hist))
    size_kb   = file_size / 1024.0
    log_size  = math.log2(max(size_kb, 0.001))

    probe = data[:65536]
    try:
        import lz4.frame
        probe_ratio = len(probe) / max(len(lz4.frame.compress(probe)), 1)
    except:
        probe_ratio = 1.0

    byte_coverage  = len(set(data[:65536])) / 256.0
    low_byte_ratio = sum(1 for b in data[:4096] if b < 128) / max(len(data[:4096]), 1)
    null_ratio     = data[:4096].count(0) / max(len(data[:4096]), 1)

    chunk  = data[:8192]
    ngrams = [chunk[i:i+4] for i in range(0, len(chunk)-4, 4)]
    repetition = sum(c-1 for c in Counter(ngrams).values()) / max(len(ngrams), 1)

    text_exts   = {'.txt','.py','.js','.ts','.json','.html','.css','.md',
                   '.csv','.log','.xml','.yaml','.yml','.ini','.cfg','.d.ts',
                   '.jsx','.tsx','.vue','.svelte','.php','.rb','.go','.rs','.c','.cpp','.h'}
    binary_exts = {'.bin','.so','.dll','.exe','.o','.a','.db','.sqlite','.wasm','.pyc'}
    media_exts  = {'.jpg','.jpeg','.png','.gif','.mp3','.mp4','.zip','.gz',
                   '.webp','.bmp','.ico','.tiff','.mov','.avi','.mkv','.flac',
                   '.woff','.woff2','.ttf','.otf','.pdf'}
    if ext in text_exts:     ext_class = 0.0
    elif ext in binary_exts: ext_class = 1.0
    elif ext in media_exts:  ext_class = 2.0
    else:                    ext_class = 0.5

    bigrams    = [chunk[i:i+2] for i in range(len(chunk)-1)]
    bigram_ent = shannon_entropy(bigrams) if bigrams else 0.0

    printable_ratio  = sum(1 for b in data[:4096] if 32 <= b <= 126) / max(len(data[:4096]), 1)
    whitespace_ratio = sum(1 for b in data[:4096] if b in (9,10,13,32))  / max(len(data[:4096]), 1)

    runs, prev, run_len = 0, (data[0] if data else 0), 1
    for b in data[1:2048]:
        if b == prev: run_len += 1
        else:
            if run_len >= 4: runs += 1
            run_len, prev = 1, b
    run_score = runs / max(len(data[:2048]), 1)

    return [
        ent, byte_std, byte_mean, byte_max,
        size_kb, log_size, probe_ratio,
        byte_coverage, low_byte_ratio, null_ratio,
        repetition, ext_class,
        bigram_ent, printable_ratio, whitespace_ratio, run_score
    ]

FEATURE_COLS = [
    'entropy','byte_std','byte_mean','byte_max',
    'size_kb','log_size','probe_ratio',
    'byte_coverage','low_byte_ratio','null_ratio',
    'repetition','ext_class',
    'bigram_ent','printable_ratio','whitespace_ratio','run_score'
]

# ── Benchmarking ──────────────────────────────────────────────────────────────

def benchmark_file(data):
    """
    Returns:
      compressible (bool): zstd achieves >1.05x ratio
      best_algo (str):     which algo wins on ratio (lz4/zstd/brotli)
    """
    import zstandard, lz4.frame, brotli as _brotli
    results = {}
    for name, fn in [
        ('zstd',   lambda d: zstandard.ZstdCompressor(level=9).compress(d)),
        ('lz4',    lambda d: lz4.frame.compress(d, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)),
        ('brotli', lambda d: _brotli.compress(d, quality=5)),
    ]:
        try:
            results[name] = len(fn(data))
        except:
            results[name] = len(data)

    best_algo = min(results, key=results.get)
    best_size = results[best_algo]
    zstd_ratio = len(data) / max(results['zstd'], 1)

    # Compressible = any algo achieves >5% reduction
    min_size = min(results.values())
    compressible = min_size < len(data) * 0.95

    return compressible, best_algo

# ── Worker ────────────────────────────────────────────────────────────────────

def process_file(fp_str):
    fp = Path(fp_str)
    try:
        file_size = fp.stat().st_size
        if file_size == 0:
            return None
        with open(fp, 'rb') as f:
            data = f.read(min(1 << 20, file_size))  # up to 1MB sample

        ext      = fp.suffix.lower()
        features = extract_features_from_bytes(data, file_size, ext)

        # Use 64KB sample for benchmarking (fast but representative)
        sample           = data[:65536]
        compressible, best_algo = benchmark_file(sample)

        return {
            'features':     features,
            'compressible': int(compressible),   # 0 or 1
            'best_algo':    best_algo,
            'raw_sample':   sample if file_size < 16384 else None,  # for dict training
        }
    except Exception as e:
        return None

# ── Main training ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Directory to train on')
    parser.add_argument('--max-files', type=int, default=100000)
    args = parser.parse_args()

    train_dir = Path(args.dataset)
    if not train_dir.is_dir():
        print(f"Not a directory: {train_dir}"); sys.exit(1)

    os.makedirs('models', exist_ok=True)

    # ── Scan files ────────────────────────────────────────────────────────────
    print(f"Scanning {train_dir}...")
    all_files = []
    for root, _, files in os.walk(train_dir):
        for fname in files:
            fp = Path(root) / fname
            try:
                if fp.is_file() and fp.stat().st_size > 0:
                    all_files.append(str(fp))
            except OSError:
                continue

    import random
    random.shuffle(all_files)
    all_files = all_files[:args.max_files]
    print(f"Processing {len(all_files)} files...")

    # ── Extract features + labels in parallel ─────────────────────────────────
    t0       = time.time()
    records  = []
    dict_samples = []
    done     = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futs = {pool.submit(process_file, fp): fp for fp in all_files}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                records.append(r)
                if r['raw_sample'] and len(dict_samples) < 300:
                    dict_samples.append(r['raw_sample'])
            done += 1
            if done % 1000 == 0 or done == len(all_files):
                print(f"   {done}/{len(all_files)} files  ({len(records)} valid)", end='\r')

    print(f"\nExtracted {len(records)} records in {time.time()-t0:.1f}s")

    if len(records) < 100:
        print("Not enough data."); sys.exit(1)

    import pandas as pd
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    X = pd.DataFrame([r['features'] for r in records], columns=FEATURE_COLS)

    # ── Model 1: Compressibility classifier ──────────────────────────────────
    print("\n" + "="*50)
    print("Training Model 1: Compressibility (binary)")
    print("="*50)

    y_comp = np.array([r['compressible'] for r in records])
    comp_ratio = y_comp.mean()
    print(f"   Compressible: {comp_ratio*100:.1f}%  Incompressible: {(1-comp_ratio)*100:.1f}%")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_comp, test_size=0.15,
                                                random_state=42, stratify=y_comp)

    comp_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.75,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42,
    )
    comp_model.fit(X_tr, y_tr,
                   eval_set=[(X_te, y_te)],
                   verbose=False)

    y_pred = comp_model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    print(f"   Accuracy: {acc*100:.1f}%")
    print(classification_report(y_te, y_pred,
                                  target_names=['incompressible','compressible'],
                                  digits=3))

    comp_model.get_booster().save_model('models/compressibility_model.json')
    print("   Saved -> models/compressibility_model.json")

    # ── Model 2: Algorithm selector (compressible files only) ─────────────────
    print("\n" + "="*50)
    print("Training Model 2: Algorithm selector (multiclass)")
    print("="*50)

    comp_records = [r for r in records if r['compressible']]
    print(f"   Using {len(comp_records)} compressible files")

    algo_labels = sorted(set(r['best_algo'] for r in comp_records))
    algo_to_int = {a: i for i, a in enumerate(algo_labels)}
    int_to_algo = {i: a for a, i in algo_to_int.items()}

    dist = Counter(r['best_algo'] for r in comp_records)
    for a, c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"   {a}: {c} ({c/len(comp_records)*100:.1f}%)")

    X2 = pd.DataFrame([r['features'] for r in comp_records], columns=FEATURE_COLS)
    y2 = np.array([algo_to_int[r['best_algo']] for r in comp_records])

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.15,
                                                    random_state=42, stratify=y2)

    algo_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.75,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
    )
    algo_model.fit(X_tr2, y_tr2,
                   eval_set=[(X_te2, y_te2)],
                   verbose=False)

    y_pred2 = algo_model.predict(X_te2)
    acc2    = accuracy_score(y_te2, y_pred2)
    print(f"   Accuracy: {acc2*100:.1f}%")
    print(classification_report(y_te2, y_pred2,
                                  target_names=[int_to_algo[i] for i in range(len(algo_labels))],
                                  digits=3))

    algo_model.get_booster().save_model('models/compressor_model.json')
    with open('models/label_map.json', 'w') as f:
        json.dump(int_to_algo, f)
    print("   Saved -> models/compressor_model.json + label_map.json")

    # ── Feature importances ───────────────────────────────────────────────────
    print("\nTop features (compressibility model):")
    imp = sorted(zip(FEATURE_COLS, comp_model.feature_importances_),
                 key=lambda x: -x[1])
    for name, score in imp[:8]:
        bar = '█' * int(score * 300)
        print(f"   {name:<20} {bar} {score:.3f}")

    # ── zstd dictionary ───────────────────────────────────────────────────────
    if dict_samples:
        print(f"\nTraining zstd dictionary on {len(dict_samples)} small-file samples...")
        try:
            import zstandard
            dict_size = 112 * 1024
            zdict = zstandard.train_dictionary(dict_size, dict_samples)
            with open('models/zstd_dict.bin', 'wb') as f:
                f.write(zdict.as_bytes())
            print(f"   Saved -> models/zstd_dict.bin ({dict_size//1024}KB)")
        except Exception as e:
            print(f"   Dictionary training failed: {e}")

    print(f"\nAll models saved to models/")
    print(f"   models/compressibility_model.json  — routes incompressible files to raw storage")
    print(f"   models/compressor_model.json       — sorts compressible files by type for better solid stream")
    print(f"   models/label_map.json")
    print(f"   models/zstd_dict.bin")

if __name__ == '__main__':
    main()

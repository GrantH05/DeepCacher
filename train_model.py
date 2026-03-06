import os
import json
import argparse
import pandas as pd
import numpy as np
import math
import time
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from pathlib import Path
import lz4.frame
import zstandard as zstd
import brotli

# ── Feature extraction ────────────────────────────────────────────────────────

def shannon_entropy(data):
    if not data: return 0.0
    freq = Counter(data)
    n = len(data)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def extract_features(file_path):
    """
    Rich feature set — 12 features instead of 3.
    More signal = model can distinguish file types far better.
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return None, None

        with open(file_path, 'rb') as f:
            raw = f.read(min(1<<20, file_size))  # up to 1MB sample

        data = raw

        # 1. Shannon entropy (overall randomness)
        ent = shannon_entropy(data)

        # 2. Byte frequency histogram stats
        hist, _ = np.histogram(list(data), bins=256, density=True)
        byte_std  = np.std(hist)
        byte_mean = np.mean(hist)
        byte_max  = np.max(hist)        # dominance of most common byte

        # 3. File size (log scale — compressors behave differently at different sizes)
        size_kb  = file_size / 1024.0
        log_size = math.log2(max(size_kb, 0.001))

        # 4. Compressibility probe — compress first 64KB with fast lz4
        #    ratio here is a direct signal of how compressible the data is
        probe = data[:65536]
        try:
            probe_compressed = lz4.frame.compress(probe)
            probe_ratio = len(probe) / max(len(probe_compressed), 1)
        except Exception:
            probe_ratio = 1.0

        # 5. Byte range coverage — how many of 256 possible byte values appear
        unique_bytes = len(set(data[:65536]))
        byte_coverage = unique_bytes / 256.0

        # 6. Low-byte ratio — text files have many bytes < 128 (ASCII)
        low_bytes = sum(1 for b in data[:4096] if b < 128)
        low_byte_ratio = low_bytes / max(len(data[:4096]), 1)

        # 7. Null byte ratio — binary files often have null bytes
        null_ratio = data[:4096].count(0) / max(len(data[:4096]), 1)

        # 8. Repetition score — count repeated 4-byte patterns in sample
        chunk = data[:8192]
        ngrams = [chunk[i:i+4] for i in range(0, len(chunk)-4, 4)]
        ngram_counts = Counter(ngrams)
        repetition = sum(c-1 for c in ngram_counts.values()) / max(len(ngrams), 1)

        # 9. File extension class (numeric encoding)
        ext = Path(file_path).suffix.lower()
        text_exts   = {'.txt','.py','.js','.ts','.json','.html','.css','.md',
                       '.csv','.log','.xml','.yaml','.yml','.ini','.cfg','.d.ts'}
        binary_exts = {'.bin','.so','.dll','.exe','.o','.a','.db','.sqlite'}
        media_exts  = {'.jpg','.jpeg','.png','.gif','.mp3','.mp4','.zip','.gz'}
        if ext in text_exts:     ext_class = 0.0
        elif ext in binary_exts: ext_class = 1.0
        elif ext in media_exts:  ext_class = 2.0
        else:                    ext_class = 0.5

        # 10. Bigram entropy — brotli excels on structured text with repeated
        #     word-level patterns; higher bigram entropy = less structure = zstd wins
        chunk2 = data[:8192]
        bigrams = [chunk2[i:i+2] for i in range(len(chunk2)-1)]
        bigram_ent = shannon_entropy(bigrams) if bigrams else 0.0

        # 11. Printable ASCII ratio — brotli is optimised for human-readable text
        printable = sum(1 for b in data[:4096] if 32 <= b <= 126)
        printable_ratio = printable / max(len(data[:4096]), 1)

        # 12. Whitespace ratio — source code / markup has lots of spaces, tabs, newlines
        whitespace = sum(1 for b in data[:4096] if b in (9, 10, 13, 32))
        whitespace_ratio = whitespace / max(len(data[:4096]), 1)

        # 13. Runs score — long runs of same byte = binary/sparse data, lz4 wins
        runs = 0
        prev = data[0] if data else 0
        run_len = 1
        for b in data[1:2048]:
            if b == prev:
                run_len += 1
            else:
                if run_len >= 4:
                    runs += 1
                run_len = 1
                prev = b
        run_score = runs / max(len(data[:2048]), 1)

        features = [
            ent, byte_std, byte_mean, byte_max,
            size_kb, log_size, probe_ratio,
            byte_coverage, low_byte_ratio, null_ratio,
            repetition, ext_class,
            bigram_ent, printable_ratio, whitespace_ratio, run_score
        ]
        return features, raw

    except Exception:
        return None, None

FEATURE_NAMES = [
    'entropy', 'byte_std', 'byte_mean', 'byte_max',
    'size_kb', 'log_size', 'probe_ratio',
    'byte_coverage', 'low_byte_ratio', 'null_ratio',
    'repetition', 'ext_class',
    'bigram_ent', 'printable_ratio', 'whitespace_ratio', 'run_score'
]

# ── Real benchmarking ─────────────────────────────────────────────────────────

def benchmark_file(data):
    """
    Label = whichever algo produces the smallest compressed output.
    Speed is NOT factored into labels — the entropy gate in _compress_one
    already handles speed at runtime. Mixing speed into labels causes
    training noise because timing varies per machine.
    """
    candidates = {
        'lz4':    lambda d: lz4.frame.compress(d, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC),
        'zstd':   lambda d: zstd.ZstdCompressor(level=6).compress(d),
        'brotli': lambda d: brotli.compress(d, quality=5),
    }
    best_name = 'zstd'   # fallback
    best_size = float('inf')
    for name, fn in candidates.items():
        try:
            size = len(fn(data))
            if size < best_size:
                best_size = size
                best_name = name
        except Exception:
            continue
    return best_name

# ── File scanner ──────────────────────────────────────────────────────────────

def scan_dataset(source_path):
    source_path = Path(source_path)
    all_files   = []
    print(f"🔍 Scanning {source_path}...")
    skip_reasons = Counter()
    for root, dirs, files in os.walk(source_path, onerror=lambda e: None):
        for file in files:
            file_path = Path(root) / file
            try:
                if not file_path.is_file():
                    skip_reasons['not_a_file'] += 1
                    continue
                file_size = file_path.stat().st_size
            except (OSError, FileNotFoundError) as e:
                skip_reasons['os_error'] += 1
                continue
            if file_size == 0:
                skip_reasons['empty'] += 1
                continue
            if file_path.suffix.lower() in {'.zip','.gz','.bz2','.xz','.7z',
                                             '.rar','.tar','.deepcacher','.pyc','.cache'}:
                skip_reasons['compressed_ext'] += 1
                continue
            all_files.append(file_path)
    print(f"✅ Found {len(all_files)} files")
    if any(skip_reasons.values()):
        print(f"⚠️  Skipped {sum(skip_reasons.values())} files:")
        for reason, count in skip_reasons.most_common():
            print(f"      {reason}: {count}")
    return all_files

# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='DeepCacher: Train compression model')
parser.add_argument('dataset', help='Directory to train on')
args = parser.parse_args()

train_dir = Path(args.dataset)
if not train_dir.is_dir():
    print(f"❌ Not a valid directory: {train_dir}")
    exit(1)

os.makedirs('models', exist_ok=True)
compressor_names = ['lz4', 'zstd', 'brotli']

print("🚀 DeepCacher Training")
print("=" * 50)
print(f"📁 Dataset: {train_dir.resolve()}")

dataset_files = scan_dataset(train_dir)
if not dataset_files:
    print("❌ No files found.")
    exit(1)

# ── Extract features + benchmark ─────────────────────────────────────────────

print(f"\n🔬 Extracting features + benchmarking {len(dataset_files)} files...")
results      = []
dict_samples = []
label_counts = Counter()

feat_skip = 0
for i, file_path in enumerate(dataset_files, 1):
    feats, raw_data = extract_features(file_path)
    if feats is None or raw_data is None:
        feat_skip += 1
        continue

    best_algo = benchmark_file(raw_data)
    best_idx  = compressor_names.index(best_algo)

    results.append(feats + [best_idx])
    label_counts[best_algo] += 1

    if len(dict_samples) < 200:
        dict_samples.append(raw_data[:16384])

    if i % 500 == 0 or i == len(dataset_files):
        print(f"   {i}/{len(dataset_files)} files processed...", end='\r')

print()
if feat_skip:
    print(f"⚠️  {feat_skip} files skipped during feature extraction (unreadable/corrupt)")

if len(results) < 5:
    print("❌ Need 5+ valid files for training!")
    exit(1)

print(f"\n📊 Label distribution: {dict(label_counts)}")

# Warn if heavily imbalanced — model will be biased toward majority class
total = sum(label_counts.values())
for name, count in label_counts.items():
    pct = count / total * 100
    if pct > 70:
        print(f"⚠️  '{name}' dominates at {pct:.0f}% — consider a more diverse dataset")

# ── Train model ───────────────────────────────────────────────────────────────

print(f"\n🤖 Training XGBoost...")
df = pd.DataFrame(results, columns=FEATURE_NAMES + ['best_algo'])
X  = df[FEATURE_NAMES]
y  = df['best_algo']

# Compute class weights to handle imbalance
class_counts = y.value_counts()
scale_pos    = {cls: total / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.75,
    min_child_weight=2,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)

print(f"\n✅ Accuracy: {acc:.1%}")
print("\n📋 Per-class breakdown:")
print(classification_report(y_test, preds, target_names=compressor_names))

# Feature importance
importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                     key=lambda x: x[1], reverse=True)
print("🔑 Top features:")
for name, imp in importances[:6]:
    bar = '█' * int(imp * 40)
    print(f"   {name:15s} {bar} {imp:.3f}")

# ── Save model ────────────────────────────────────────────────────────────────

label_map = {str(i): name for i, name in enumerate(compressor_names)}
with open('models/label_map.json', 'w') as f:
    json.dump(label_map, f)
model.save_model('models/compressor_model.json')
print(f"\n💾 Model saved → models/compressor_model.json")

# ── Train zstd dictionary ─────────────────────────────────────────────────────

print(f"\n📚 Training zstd dictionary on {len(dict_samples)} samples...")
try:
    dict_size = 112 * 1024
    zstd_dict = zstd.train_dictionary(dict_size, dict_samples)
    dict_data = zstd_dict.as_bytes()
    with open('models/zstd_dict.bin', 'wb') as f:
        f.write(dict_data)
    print(f"✅ Dictionary saved: {len(dict_data)/1024:.1f} KB → models/zstd_dict.bin")
except Exception as e:
    print(f"⚠️  Dictionary training skipped: {e}")

print("\n🎉 TRAINING COMPLETE!")
print(f"   Files:    {len(results)}")
print(f"   Accuracy: {acc:.1%}")
print("\n🚀 Run: python3 compressor.py your_folder/")

import argparse
import json
import struct
import time
import zlib
import math
import numpy as np
import lz4.frame
import zstandard as zstd
import brotli
import os
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from xgboost import Booster, DMatrix
import pandas as pd
from pathlib import Path
import shutil

def shannon_entropy(data):
    if not data: return 0.0
    freq = Counter(data)
    n = len(data)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def extract_features(file_path):
    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        data = f.read(min(1<<20, file_size))

    ent      = shannon_entropy(data)
    hist, _  = np.histogram(list(data), bins=256, density=True)
    byte_std  = np.std(hist)
    byte_mean = np.mean(hist)
    byte_max  = np.max(hist)
    size_kb   = file_size / 1024.0
    log_size  = math.log2(max(size_kb, 0.001))

    probe = data[:65536]
    try:
        probe_ratio = len(probe) / max(len(lz4.frame.compress(probe)), 1)
    except Exception:
        probe_ratio = 1.0

    unique_bytes   = len(set(data[:65536]))
    byte_coverage  = unique_bytes / 256.0
    low_byte_ratio = sum(1 for b in data[:4096] if b < 128) / max(len(data[:4096]), 1)
    null_ratio     = data[:4096].count(0) / max(len(data[:4096]), 1)

    chunk  = data[:8192]
    ngrams = [chunk[i:i+4] for i in range(0, len(chunk)-4, 4)]
    repetition = sum(c-1 for c in Counter(ngrams).values()) / max(len(ngrams), 1)

    ext = Path(file_path).suffix.lower()
    text_exts   = {'.txt','.py','.js','.ts','.json','.html','.css','.md',
                   '.csv','.log','.xml','.yaml','.yml','.ini','.cfg','.d.ts'}
    binary_exts = {'.bin','.so','.dll','.exe','.o','.a','.db','.sqlite'}
    media_exts  = {'.jpg','.jpeg','.png','.gif','.mp3','.mp4','.zip','.gz'}
    if ext in text_exts:     ext_class = 0.0
    elif ext in binary_exts: ext_class = 1.0
    elif ext in media_exts:  ext_class = 2.0
    else:                    ext_class = 0.5

    chunk2    = data[:8192]
    bigrams   = [chunk2[i:i+2] for i in range(len(chunk2)-1)]
    bigram_ent = shannon_entropy(bigrams) if bigrams else 0.0

    printable_ratio  = sum(1 for b in data[:4096] if 32 <= b <= 126) / max(len(data[:4096]), 1)
    whitespace_ratio = sum(1 for b in data[:4096] if b in (9,10,13,32)) / max(len(data[:4096]), 1)

    runs, prev, run_len = 0, (data[0] if data else 0), 1
    for b in data[1:2048]:
        if b == prev: run_len += 1
        else:
            if run_len >= 4: runs += 1
            run_len, prev = 1, b
    run_score = runs / max(len(data[:2048]), 1)

    features = [
        ent, byte_std, byte_mean, byte_max,
        size_kb, log_size, probe_ratio,
        byte_coverage, low_byte_ratio, null_ratio,
        repetition, ext_class,
        bigram_ent, printable_ratio, whitespace_ratio, run_score
    ]
    cols = [
        'entropy','byte_std','byte_mean','byte_max',
        'size_kb','log_size','probe_ratio',
        'byte_coverage','low_byte_ratio','null_ratio',
        'repetition','ext_class',
        'bigram_ent','printable_ratio','whitespace_ratio','run_score'
    ]
    return np.array([features]), cols

# Load model + safety checks
for path in ['models/compressor_model.json', 'models/label_map.json']:
    if not os.path.exists(path):
        print(f"Missing {path}. Run: python3 train_model.py")
        exit(1)

with open('models/label_map.json', 'r') as f:
    label_map = json.load(f)

model = Booster()
model.load_model('models/compressor_model.json')

# Load zstd dictionary if available (trained by train_model.py)
# Dramatically improves ratio on small similar files (JSON, TS, configs)
_zstd_dict = None
if os.path.exists('models/zstd_dict.bin'):
    with open('models/zstd_dict.bin', 'rb') as f:
        _zstd_dict = zstd.ZstdCompressionDict(f.read())
    print("📚 zstd dictionary loaded")

compressors = {
    'lz4':    lambda d: lz4.frame.compress(d, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC),
    'zstd':   lambda d: zstd.ZstdCompressor(level=6, dict_data=_zstd_dict).compress(d),
    'brotli': lambda d: brotli.compress(d, quality=5),
}

def predict_best(file_path):
    feats, cols = extract_features(file_path)
    feat_df = pd.DataFrame(feats, columns=cols)
    dmat = DMatrix(feat_df)
    pred_probs = model.predict(dmat)
    algo_names = list(label_map.values())
    ranked = sorted(range(len(pred_probs[0])), key=lambda i: pred_probs[0][i], reverse=True)
    top2 = [algo_names[i] for i in ranked[:2]]
    sample_size = min(65536, os.path.getsize(file_path))
    with open(file_path, 'rb') as f:
        sample = f.read(sample_size)
    best_algo = None
    best_size = float('inf')
    for algo in top2:
        try:
            size = len(compressors[algo](sample))
            if size < best_size:
                best_size = size
                best_algo = algo
        except Exception:
            continue
    return best_algo or top2[0]

MAGIC = b'DCACHE\x01\x00'

# Top-level function required for ProcessPoolExecutor pickling.
# All imports are local — each worker process is fully self-contained.
# CRC32 is computed inside worker so raw bytes never travel back to main process.
# Module-level cache inside each worker process.
# ProcessPoolExecutor creates N worker processes; each loads the dict once
# on first call and reuses it for all subsequent files — zero extra disk reads.
_worker_zstd_dict  = None
_worker_dict_loaded = False

def _get_worker_dict():
    global _worker_zstd_dict, _worker_dict_loaded
    if not _worker_dict_loaded:
        import zstandard as _zstd
        if os.path.exists('models/zstd_dict.bin'):
            with open('models/zstd_dict.bin', 'rb') as f:
                _worker_zstd_dict = _zstd.ZstdCompressionDict(f.read())
        _worker_dict_loaded = True
    return _worker_zstd_dict

def _compress_one(args):
    import zlib as _zlib, math as _math
    import lz4.frame as _lz4
    import zstandard as _zstd
    import brotli as _brotli
    from collections import Counter
    from pathlib import Path

    file_path_str, folder_root_str, algo = args
    file_path   = Path(file_path_str)
    folder_root = Path(folder_root_str)

    with open(file_path, 'rb') as f:
        raw = f.read()

    orig_size = len(raw)
    crc32     = _zlib.crc32(raw) & 0xFFFFFFFF

    sample = raw[:65536]
    if sample:
        freq = Counter(sample)
        n    = len(sample)
        ent  = -sum((c/n)*_math.log2(c/n) for c in freq.values())
    else:
        ent = 0.0

    # Load dict once per worker process, reuse for every file
    d = _get_worker_dict()

    _comp = {
        'lz4':    lambda d_: _lz4.compress(d_, compression_level=_lz4.COMPRESSIONLEVEL_MINHC),
        'zstd':   lambda d_: _zstd.ZstdCompressor(level=6, dict_data=d).compress(d_),
        'brotli': lambda d_: _brotli.compress(d_, quality=5),
    }

    if ent > 7.2:
        candidates = ['lz4']
    elif ent < 4.0:
        candidates = ['zstd', 'brotli']
    else:
        candidates = list({algo, 'zstd'})

    best_algo       = None
    best_compressed = None

    for candidate in candidates:
        try:
            compressed = _comp[candidate](raw)
            if best_compressed is None or len(compressed) < len(best_compressed):
                best_compressed = compressed
                best_algo       = candidate
        except Exception:
            continue

    if best_compressed is None:
        best_algo       = 'zstd'
        best_compressed = _zstd.ZstdCompressor(level=3, dict_data=d).compress(raw)

    arcname = str(file_path.relative_to(folder_root))
    return arcname, best_algo, best_compressed, orig_size, crc32

def _write_archive(output_path, entries):
    index      = []
    data_parts = []
    offset     = 0
    for e in entries:
        comp_size = len(e['data'])
        index.append({
            'name':      e['name'],
            'algo':      e['algo'],
            'offset':    offset,
            'comp_size': comp_size,
            'orig_size': e['orig_size'],
            'crc32':     e['crc32'],
        })
        data_parts.append(e['data'])
        offset += comp_size

    index_json = json.dumps(index, separators=(',', ':')).encode()
    index_len  = struct.pack('<I', len(index_json))

    # Embed dictionary so archive is fully self-contained.
    # Without this, retraining overwrites models/zstd_dict.bin and old
    # archives become undecompressable (dictionary mismatch error).
    dict_bytes = b''
    if _zstd_dict is not None:
        dict_bytes = _zstd_dict.as_bytes()
    dict_len = struct.pack('<I', len(dict_bytes))

    with open(output_path, 'wb') as f:
        f.write(MAGIC)
        f.write(dict_len)       # 4 bytes: dictionary length (0 = no dict)
        f.write(dict_bytes)     # N bytes: raw dictionary (may be empty)
        f.write(index_len)      # 4 bytes: index length
        f.write(index_json)     # index JSON
        for blob in data_parts:
            f.write(blob)

def compress_folder(folder_path, output_path):
    folder_path = Path(folder_path)
    output_path = Path(output_path)

    if not folder_path.is_dir():
        print(f"Not a folder: {folder_path}")
        return None, None

    print(f"\nANALYZING FOLDER: {folder_path.name}")
    print("\nSCANNING ALL FILES...")

    all_files = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            fp = Path(root) / fname
            try:
                if fp.is_file():
                    all_files.append(fp)
            except OSError:
                continue

    if not all_files:
        print("No files found.")
        return None, None

    text_exts = {'.txt','.py','.js','.ts','.json','.html','.css',
                 '.md','.csv','.log','.xml','.yaml','.yml','.ini','.cfg'}
    type_counts = {'text': 0, 'binary': 0, 'other': 0}
    for fp in all_files:
        ext = fp.suffix.lower()
        if ext in text_exts:                             type_counts['text']   += 1
        elif ext in {'.bin','.so','.dll','.exe','.o'}:  type_counts['binary'] += 1
        else:                                             type_counts['other']  += 1

    total = len(all_files)
    print(f"\nFOLDER CONTENTS ({total} files total):")
    print(f"   Text:   {type_counts['text']:4d}")
    print(f"   Binary: {type_counts['binary']:4d}")
    print(f"   Other:  {type_counts['other']:4d}")

    print("\nAI SAMPLING...")
    algo_votes = {'lz4': 0, 'zstd': 0, 'brotli': 0}
    file_algos = {}
    for i, fp in enumerate(all_files[:20], 1):
        try:
            algo = predict_best(fp)
        except Exception:
            algo = 'zstd'
        algo_votes[algo] += 1
        file_algos[fp] = algo
        print(f"   {i:2d}  {fp.name[:35]:35s} -> {algo}")

    default_algo = max(algo_votes, key=algo_votes.get)
    print(f"\nDEFAULT ALGO: {default_algo}")
    for fp in all_files:
        if fp not in file_algos:
            file_algos[fp] = default_algo

    print(f"\nCOMPRESSING {total} files (multiprocess, {os.cpu_count()} cores)...")
    t_start    = time.time()
    orig_total = 0
    comp_total = 0
    done       = 0
    entries    = [None] * total
    fp_index   = {fp: i for i, fp in enumerate(all_files)}

    worker_args = [
        (str(fp), str(folder_path), file_algos[fp])
        for fp in all_files
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        futures = {
            pool.submit(_compress_one, arg): Path(arg[0])
            for arg in worker_args
        }
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                arcname, algo, comp_data, orig_size, crc32 = fut.result()
                idx = fp_index[fp]
                entries[idx] = {
                    'name':      arcname,
                    'algo':      algo,
                    'data':      comp_data,
                    'orig_size': orig_size,
                    'crc32':     crc32,
                }
                orig_total += orig_size
                comp_total += len(comp_data)
            except Exception as exc:
                print(f"\n   WARNING: Skipping {fp.name}: {exc}")
                entries[fp_index[fp]] = None

            done += 1
            if done % 10 == 0 or done == total:
                pct = done / total * 100
                print(f"   {done}/{total} ({pct:.0f}%)  "
                      f"{comp_total/1024/1024:.1f} MB compressed", end='\r')

    entries = [e for e in entries if e is not None]
    print()

    print(f"\nWRITING ARCHIVE -> {output_path.name} ...")
    _write_archive(output_path, entries)

    elapsed = time.time() - t_start
    ratio   = orig_total / comp_total if comp_total else 1.0
    saved   = (orig_total - comp_total) / 1024 / 1024

    print(f"\nDONE in {elapsed:.1f}s")
    print(f"   Original:   {orig_total/1024/1024:.1f} MB")
    print(f"   Compressed: {comp_total/1024/1024:.1f} MB")
    print(f"   Saved:      {saved:.1f} MB  ({ratio:.2f}x ratio)")
    print(f"   Output:     {output_path}")

    return default_algo, ratio

def decompress_folder(archive_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, 'rb') as f:
        header = f.read(8)

    if header == MAGIC:
        with open(archive_path, 'rb') as f:
            f.read(8)                                          # skip magic
            dict_len   = struct.unpack('<I', f.read(4))[0]    # embedded dict size
            dict_bytes = f.read(dict_len)                      # embedded dict data
            idx_len    = struct.unpack('<I', f.read(4))[0]
            index      = json.loads(f.read(idx_len))
            data_block = f.read()

        # Use the dictionary that was embedded at compression time —
        # completely independent of whatever is in models/zstd_dict.bin now
        if dict_bytes:
            embedded_dict = zstd.ZstdCompressionDict(dict_bytes)
            zstd_decomp   = lambda d: zstd.ZstdDecompressor(dict_data=embedded_dict).decompress(d)
        else:
            zstd_decomp   = lambda d: zstd.ZstdDecompressor().decompress(d)

        decompressors = {
            'lz4':    lz4.frame.decompress,
            'zstd':   zstd_decomp,
            'brotli': brotli.decompress,
        }
        print(f"Extracting {len(index)} files -> {output_dir}")
        errors = []
        for i, entry in enumerate(index, 1):
            blob = data_block[entry['offset'] : entry['offset'] + entry['comp_size']]
            try:
                raw = decompressors[entry['algo']](blob)
            except Exception as e:
                errors.append(f"DECOMPRESS FAIL: {entry['name']} ({e})")
                continue

            if len(raw) != entry['orig_size']:
                errors.append(
                    f"SIZE MISMATCH: {entry['name']} "
                    f"(expected {entry['orig_size']}B, got {len(raw)}B)"
                )
                continue

            if 'crc32' in entry:
                actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
                if actual_crc != entry['crc32']:
                    errors.append(
                        f"CORRUPTION: {entry['name']} "
                        f"(expected {entry['crc32']:#010x}, got {actual_crc:#010x})"
                    )
                    continue

            out = output_dir / entry['name']
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'wb') as f:
                f.write(raw)

            if i % 20 == 0 or i == len(index):
                print(f"   {i}/{len(index)} files", end='\r')

        print()
        if errors:
            print(f"\nINTEGRITY ERRORS ({len(errors)}):")
            for e in errors:
                print(f"   - {e}")
        else:
            print(f"All {len(index)} files verified and extracted -> {output_dir}")
        return

    # Legacy TAR fallback
    print("Legacy archive, trying TAR fallback...")
    with open(archive_path, 'rb') as f:
        data = f.read()

    tar_data = None
    for name, fn in [
        ('lz4',    lz4.frame.decompress),
        ('zstd',   lambda d: zstd.ZstdDecompressor().decompress(d)),
        ('brotli', brotli.decompress),
    ]:
        try:
            tar_data = fn(data)
            print(f"Detected {name} (legacy)")
            break
        except Exception:
            continue

    if tar_data is None:
        print("Unknown compression format")
        return

    temp_tar = output_dir / "_temp_legacy.tar"
    with open(temp_tar, 'wb') as f:
        f.write(tar_data)
    tar = tarfile.open(temp_tar, 'r:*')
    try:
        tar.extractall(output_dir)
        print(f"Legacy extraction complete -> {output_dir}")
    finally:
        tar.close()
        if temp_tar.exists():
            temp_tar.unlink()

def main():
    parser = argparse.ArgumentParser(description='DeepCacher: AI File/Folder Compressor')
    parser.add_argument('input', help='File or folder to compress')
    parser.add_argument('--output', '-o', help='Output file (auto: input.deepcacher)')
    parser.add_argument('--decompress', '-d', action='store_true', help='Decompress mode')
    parser.add_argument('--benchmark', '-b', action='store_true')
    args = parser.parse_args()

    if args.decompress:
        output_dir = Path(args.output or args.input.replace('.deepcacher', '_extracted'))
        decompress_folder(args.input, output_dir)
        return

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix('.deepcacher')
    os.makedirs('outputs', exist_ok=True)
    output_path = Path('outputs') / output_path.name

    if input_path.is_file():
        orig_size = input_path.stat().st_size
        print(f"Analyzing {input_path.name} ({orig_size/1024:.1f}KB)...")
        with open(input_path, 'rb') as f:
            data = f.read()
        best_algo  = predict_best(input_path)
        compressed = compressors[best_algo](data)
        ratio      = orig_size / len(compressed)
        with open(output_path, 'wb') as f:
            f.write(compressed)
        print(f"Compressed '{input_path.name}'")
        print(f"   Algo: {best_algo} | Ratio: {ratio:.2f}x")

    elif input_path.is_dir():
        best_algo, ratio = compress_folder(input_path, output_path)
        if best_algo:
            print(f"\nFOLDER COMPRESSED!")
            print(f"   Folder: {input_path.name}")
            print(f"   Algo: {best_algo} | Ratio: {ratio:.2f}x")
            print(f"   Output: {output_path}")
    else:
        print(f"Invalid path: {input_path}")

if __name__ == '__main__':
    main()

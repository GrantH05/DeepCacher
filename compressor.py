#!/usr/bin/env python3
"""
DeepCacher — AI-powered folder compressor

Architecture:
  - Model 1 (compressibility): routes incompressible files to raw/lz4 storage
    instead of wasting CPU trying to compress them through the solid stream
  - Model 2 (algo selector): sorts files in the solid stream by predicted type
    (lz4-class / zstd-class / brotli-class) so similar files are adjacent,
    improving zstd's cross-file pattern matching within its sliding window
  - Global solid zstd stream: all compressible files in one continuous stream
  - zstd dictionary: pre-trained patterns for small similar files
"""

import argparse, json, struct, time, zlib, math, os, tarfile, sys
import numpy as np
import lz4.frame
import zstandard as zstd
import brotli
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from xgboost import Booster, DMatrix
import pandas as pd
from pathlib import Path

# ── Feature extraction ────────────────────────────────────────────────────────

# Extension lookup tables (set membership is O(1))
_TEXT_EXTS = {'.txt','.py','.js','.ts','.json','.html','.css','.md',
              '.csv','.log','.xml','.yaml','.yml','.ini','.cfg','.d.ts',
              '.jsx','.tsx','.vue','.svelte','.php','.rb','.go','.rs','.c','.cpp','.h'}
_BIN_EXTS  = {'.bin','.so','.dll','.exe','.o','.a','.db','.sqlite','.wasm','.pyc'}
_MEDIA_EXTS= {'.jpg','.jpeg','.png','.gif','.mp3','.mp4','.zip','.gz',
              '.webp','.bmp','.ico','.tiff','.mov','.avi','.mkv','.flac',
              '.woff','.woff2','.ttf','.otf','.pdf'}

def extract_features(data, file_size, ext, precomputed_probe=None):
    """Fast vectorized feature extraction — avoids Python loops where possible."""
    arr65  = np.frombuffer(data[:min(65536, len(data))], dtype=np.uint8)
    arr4k  = np.frombuffer(data[:4096],  dtype=np.uint8) if len(data) >= 4 else arr65[:4]
    arr8k  = np.frombuffer(data[:8192],  dtype=np.uint8) if len(data) >= 8 else arr65[:8]
    arr2k  = np.frombuffer(data[:2048],  dtype=np.uint8) if len(data) >= 2 else arr65[:2]

    # Byte histogram — np.bincount is 10x faster than np.histogram on integer arrays
    hist = np.bincount(arr65, minlength=256).astype(np.float64)
    hist_norm = hist / max(hist.sum(), 1)

    # Entropy from histogram (vectorized, no Python loop)
    nz  = hist_norm[hist_norm > 0]
    ent = float(-np.sum(nz * np.log2(nz)))

    byte_std  = float(np.std(hist_norm))
    byte_mean = float(np.mean(hist_norm))
    byte_max  = float(np.max(hist_norm))

    size_kb  = file_size / 1024.0
    log_size = math.log2(max(size_kb, 0.001))

    # lz4 probe — still needs actual compression, cap at 16KB for speed
    if precomputed_probe is not None:
        probe_ratio = precomputed_probe
    else:
        probe = data[:16384]
        try:    probe_ratio = len(probe) / max(len(lz4.frame.compress(probe)), 1)
        except: probe_ratio = 1.0

    byte_coverage  = float(np.count_nonzero(hist)) / 256.0
    low_byte_ratio = float(np.sum(arr4k < 128))  / max(len(arr4k), 1)
    null_ratio     = float(np.sum(arr4k == 0))   / max(len(arr4k), 1)

    # 4-gram repetition — use numpy stride tricks (no Python loop)
    if len(arr8k) >= 8:
        # Pack consecutive 4-bytes into uint32 for fast hashing
        view = np.lib.stride_tricks.as_strided(
            arr8k.view(np.uint8),
            shape=(len(arr8k)//4, 4),
            strides=(4, 1)
        )
        keys = view.view(np.uint32).ravel() if view.shape[1] == 4 else np.array([], np.uint32)
        if len(keys) > 1:
            _, counts = np.unique(keys, return_counts=True)
            repetition = float(np.sum(counts - 1)) / max(len(keys), 1)
        else:
            repetition = 0.0
    else:
        repetition = 0.0

    if ext in _TEXT_EXTS:    ext_class = 0.0
    elif ext in _BIN_EXTS:   ext_class = 1.0
    elif ext in _MEDIA_EXTS: ext_class = 2.0
    else:                    ext_class = 0.5

    # Bigram entropy — vectorized
    if len(arr8k) >= 2:
        bigrams = arr8k[:-1].astype(np.uint16) * 256 + arr8k[1:].astype(np.uint16)
        bhist   = np.bincount(bigrams, minlength=65536).astype(np.float64)
        bnz     = bhist[bhist > 0] / bhist.sum()
        bigram_ent = float(-np.sum(bnz * np.log2(bnz)))
    else:
        bigram_ent = 0.0

    printable_ratio  = float(np.sum((arr4k >= 32) & (arr4k <= 126))) / max(len(arr4k), 1)
    whitespace_ratio = float(np.sum(np.isin(arr4k, [9,10,13,32])))   / max(len(arr4k), 1)

    # Run-length score — vectorized diff
    if len(arr2k) >= 2:
        diffs    = np.diff(arr2k)
        run_ends = np.where(diffs != 0)[0]
        run_lens = np.diff(np.concatenate([[0], run_ends+1, [len(arr2k)]]))
        run_score = float(np.sum(run_lens >= 4)) / max(len(arr2k), 1)
    else:
        run_score = 0.0

    return [
        ent, byte_std, byte_mean, byte_max,
        size_kb, log_size, probe_ratio,
        byte_coverage, low_byte_ratio, null_ratio,
        repetition, ext_class,
        bigram_ent, printable_ratio, whitespace_ratio, run_score
    ]

def _extract_features_chunk(chunk):
    """Pure CPU feature extraction — no file I/O, no lz4 probe.
    Receives (fp_str, 4KB_sample, file_size, ext, probe_ratio=1.0).
    Pure numpy math — very fast, no compression calls.
    """
    results = {}
    for fp_str, sample, file_size, ext, probe_ratio in chunk:
        try:
            results[fp_str] = extract_features(sample, file_size, ext,
                                               precomputed_probe=probe_ratio)
        except Exception:
            results[fp_str] = None
    return results

FEATURE_COLS = [
    'entropy','byte_std','byte_mean','byte_max',
    'size_kb','log_size','probe_ratio',
    'byte_coverage','low_byte_ratio','null_ratio',
    'repetition','ext_class',
    'bigram_ent','printable_ratio','whitespace_ratio','run_score'
]

# ── Model loading ─────────────────────────────────────────────────────────────

def _check_models():
    missing = [p for p in [
        'models/compressibility_model.json',
        'models/compressor_model.json',
        'models/label_map.json',
    ] if not os.path.exists(p)]
    if missing:
        print(f"Missing model files: {missing}")
        print("Run: python3 train_model.py /path/to/dataset/")
        sys.exit(1)

_check_models()

with open('models/label_map.json') as f:
    _label_map = json.load(f)           # {int_str: algo_name}
_algo_names = list(_label_map.values()) # e.g. ['brotli','lz4','zstd']

_comp_model = Booster()
_comp_model.load_model('models/compressibility_model.json')

_algo_model = Booster()
_algo_model.load_model('models/compressor_model.json')

_zstd_dict = None
if os.path.exists('models/zstd_dict.bin'):
    with open('models/zstd_dict.bin', 'rb') as f:
        _zstd_dict = zstd.ZstdCompressionDict(f.read())
    print("zstd dictionary loaded")

# ── Model inference ───────────────────────────────────────────────────────────

def _predict_batch(file_data_list, sample_size=65536):
    """
    Given list of (fp, raw_bytes), returns:
      - solid_files:  [(fp, sort_key)] to go into solid zstd stream, sorted by algo group
      - skip_files:   [fp] incompressible — store raw or lz4

    Uses both models:
      Model 1 (compressibility): filters out files that won't compress
      Model 2 (algo selector):   assigns sort key so similar files are adjacent in stream
    """
    # Parallel CPU-only feature extraction.
    # Each chunk contains (fp_str, sample_bytes, file_size, ext) — ~64KB per file.
    # 8 chunks × ~5k files × 64KB = ~2.5GB... still too much.
    # Use 4KB samples for feature extraction — enough for entropy/histogram features.
    # lz4 probe uses its own 16KB internal cap anyway.
    WORKERS    = os.cpu_count() or 4
    fps        = [fp for fp, _ in file_data_list]
    exts       = [fp.suffix.lower() for fp in fps]

    # Send only 4KB sample + neutral probe_ratio=1.0 to workers.
    # lz4 probe (672MB of compression for 42k files) took 11s and is skipped.
    # The model has 15 other features (entropy, null_ratio, byte_std, etc.) that
    # are sufficient for compressibility classification. probe_ratio=1.0 is neutral.
    # IPC payload: 4KB × 42k = 164MB total.
    HIST_SAMPLE = 4096

    all_tuples = [
        (str(fp), raw[:HIST_SAMPLE], len(raw), fp.suffix.lower(), 1.0)
        for fp, raw in file_data_list
    ]
    chunk_size = max(1, math.ceil(len(all_tuples) / WORKERS))
    chunks     = [all_tuples[i:i+chunk_size] for i in range(0, len(all_tuples), chunk_size)]

    rows_map = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(_extract_features_chunk, chunk) for chunk in chunks]
        for fut in as_completed(futs):
            rows_map.update(fut.result())
    print(f"   {len(rows_map)}/{len(all_tuples)} features extracted")

    rows = [rows_map.get(str(fp)) for fp in fps]
    # Files with failed extraction default to compressible with no sort preference
    # so they go into the solid stream rather than being lost
    fallback = extract_features(b'hello world ' * 100, 1200, '.txt')
    rows = [r if r is not None else fallback for r in rows]

    X   = pd.DataFrame(rows, columns=FEATURE_COLS)
    dm  = DMatrix(X)

    # Model 1: compressibility probability
    comp_probs  = _comp_model.predict(dm)   # shape (N,) — prob of being compressible
    is_comp     = comp_probs > 0.5          # threshold

    # Model 2: algo prediction (only for compressible files, but run on all for speed)
    algo_probs  = _algo_model.predict(dm)   # shape (N, n_algos)
    # Sort key = predicted algo index → files of same predicted type end up adjacent
    if algo_probs.ndim == 1:
        algo_idx = algo_probs.astype(int)
    else:
        algo_idx = np.argmax(algo_probs, axis=1)

    solid_files = []
    skip_files  = []
    for i, fp in enumerate(fps):
        if is_comp[i]:
            # sort_key = (algo_group, ext) → same-type files cluster together
            sort_key = (int(algo_idx[i]), exts[i])
            solid_files.append((fp, sort_key))
        else:
            skip_files.append(fp)

    return solid_files, skip_files

# ── Archive format ────────────────────────────────────────────────────────────
MAGIC = b'DCACHE\x03\x00'   # v3

# ── Binary solid file index packing ──────────────────────────────────────────

def _pack_solid_files(file_entries):
    """Pack [name, raw_offset, orig_size, crc32] list as binary structs."""
    parts = []
    for name, raw_offset, orig_size, crc32 in file_entries:
        name_b = name.encode('utf-8')
        parts.append(struct.pack('<H', len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack('<QII', raw_offset, orig_size, crc32))
    return b''.join(parts)

def _unpack_solid_files(data):
    entries = []
    i = 0
    while i < len(data):
        name_len = struct.unpack_from('<H', data, i)[0]; i += 2
        name     = data[i:i+name_len].decode('utf-8');   i += name_len
        raw_offset, orig_size, crc32 = struct.unpack_from('<QII', data, i); i += 16
        entries.append([name, raw_offset, orig_size, crc32])
    return entries

# ── Archive writer ─────────────────────────────────────────────────────────────

def _write_archive(output_path, index, data_parts, solid_files_bin=b'', dict_bytes=b''):
    clean_index = []
    for entry in index:
        e = {k: v for k, v in entry.items() if k != 'files'}
        if 'files' in entry:
            e['files_bin'] = True
        clean_index.append(e)

    index_json = json.dumps(clean_index, separators=(',', ':')).encode()

    with open(output_path, 'wb', buffering=64*1024*1024) as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', len(dict_bytes)));       f.write(dict_bytes)
        f.write(struct.pack('<I', len(index_json)));       f.write(index_json)
        f.write(struct.pack('<I', len(solid_files_bin)));  f.write(solid_files_bin)
        for blob in data_parts:
            f.write(blob)

# ── Binary worker (incompressible files) ─────────────────────────────────────

def _compress_binary(args):
    """lz4 for model-predicted incompressible files. If lz4 expands, store raw."""
    import zlib as _zlib, lz4.frame as _lz4
    from pathlib import Path
    fp_str, root_str = args
    fp, root = Path(fp_str), Path(root_str)
    with open(fp, 'rb') as f:
        raw = f.read()
    crc32 = _zlib.crc32(raw) & 0xFFFFFFFF
    try:
        comp = _lz4.compress(raw, compression_level=_lz4.COMPRESSIONLEVEL_MINHC)
        algo = 'lz4' if len(comp) < len(raw) else 'raw'
        if algo == 'raw': comp = raw
    except:
        comp, algo = raw, 'raw'
    return str(fp.relative_to(root)), algo, comp, len(raw), crc32

# ── Main compress_folder ──────────────────────────────────────────────────────

def compress_folder(folder_path, output_path):
    folder_path = Path(folder_path)
    output_path = Path(output_path)

    if not folder_path.is_dir():
        print(f"Not a folder: {folder_path}"); return None, None

    print(f"\nANALYZING: {folder_path.name}")

    all_files = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            fp = Path(root) / fname
            try:
                if fp.is_file() and fp.stat().st_size > 0:
                    all_files.append(fp)
            except OSError:
                continue

    if not all_files:
        print("No files found."); return None, None

    total = len(all_files)
    print(f"{total} files found")

    # ── Step 1: Read all files in parallel (threads, I/O bound) ─────────────
    print("Reading files...")
    t0 = time.time()
    READERS = min(32, (os.cpu_count() or 4) * 2)
    file_data  = {}   # fp -> raw bytes (full content for compression)
    read_errors = []

    def _read_file(fp):
        with open(fp, 'rb') as f:
            return f.read()

    with ThreadPoolExecutor(max_workers=READERS) as pool:
        futs = {pool.submit(_read_file, fp): fp for fp in all_files}
        done = 0
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                file_data[fp] = fut.result()
            except Exception as e:
                read_errors.append(fp)
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"   {done}/{total} read", end='\r')
    print(f"\n   Read {len(file_data)} files in {time.time()-t0:.1f}s"
          + (f"  ({len(read_errors)} errors)" if read_errors else ""))

    # ── Step 2: Feature extraction (parallel CPU) + model inference ───────────
    # Pass only a 64KB sample per file to workers — not full contents.
    # Full contents stay in file_data dict in main process for the solid stream.
    # This means: no file loss (reads done once, errors tracked), no redundant I/O.
    print("Running model inference (batch)...")
    t0 = time.time()

    SAMPLE = 8192   # 8KB covers all feature slices (largest is data[:8192])
    items = [(fp, file_data[fp]) for fp in all_files if fp in file_data]
    solid_tagged, skip_files = _predict_batch(items, sample_size=SAMPLE)

    # Sort solid files: algo group first, then extension — maximises adjacency
    # of similar files so zstd sees repeated patterns across its sliding window
    solid_tagged.sort(key=lambda x: x[1])
    solid_files = [fp for fp, _ in solid_tagged]

    n_solid = len(solid_files)
    n_skip  = len(skip_files)
    print(f"   Solid stream: {n_solid} files  |  Skip (raw/lz4): {n_skip} files  "
          f"[{time.time()-t0:.1f}s]")

    t_start    = time.time()
    orig_total = 0
    comp_total = 0
    index      = []
    data_parts = []
    offset     = 0

    # ── Step 3: Global solid stream ───────────────────────────────────────────
    if solid_files:
        print(f"\nBuilding solid stream ({n_solid} files)...")
        cctx = zstd.ZstdCompressor(level=9, threads=-1, dict_data=_zstd_dict)

        solid_file_entries = []
        parts = []
        for fp in solid_files:
            raw     = file_data[fp]   # already in memory — no re-read
            arcname = str(fp.relative_to(folder_path))
            raw_len = len(raw)
            crc32   = zlib.crc32(raw) & 0xFFFFFFFF
            solid_file_entries.append([arcname, orig_total, raw_len, crc32])
            parts.append(raw)
            orig_total += raw_len

        print(f"   Concatenating {orig_total/1024/1024:.1f} MB...")
        concat = b''.join(parts)
        del parts

        print(f"   Compressing with {os.cpu_count()} threads...")
        t_c = time.time()
        solid_compressed = cctx.compress(concat)
        del concat
        print(f"   {orig_total/1024/1024:.1f} MB → {len(solid_compressed)/1024/1024:.1f} MB  "
              f"({orig_total/max(len(solid_compressed),1):.2f}×)  [{time.time()-t_c:.1f}s]")

        solid_files_bin = _pack_solid_files(solid_file_entries)
        comp_total += len(solid_compressed)

        index.append({
            'name':      '__solid_stream__',
            'algo':      'solid_stream',
            'offset':    offset,
            'comp_size': len(solid_compressed),
            'orig_size': orig_total,
            'files':     solid_file_entries,   # stripped in _write_archive, replaced by binary
        })
        data_parts.append(solid_compressed)
        offset += len(solid_compressed)
    else:
        solid_files_bin = b''

    # ── Step 4: Incompressible files (parallel lz4/raw) ───────────────────────
    if skip_files:
        print(f"\nStoring {n_skip} incompressible files...")
        worker_args = [(str(fp), str(folder_path)) for fp in skip_files]
        done = 0
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            futs = {pool.submit(_compress_binary, arg): Path(arg[0]) for arg in worker_args}
            for fut in as_completed(futs):
                try:
                    arcname, algo, comp_data, raw_size, crc32 = fut.result()
                    index.append({
                        'name':      arcname,
                        'algo':      algo,
                        'offset':    offset,
                        'comp_size': len(comp_data),
                        'orig_size': raw_size,
                        'crc32':     crc32,
                    })
                    data_parts.append(comp_data)
                    offset     += len(comp_data)
                    orig_total += raw_size
                    comp_total += len(comp_data)
                except Exception as exc:
                    print(f"   WARNING: {exc}")
                done += 1
                if done % 50 == 0 or done == n_skip:
                    print(f"   {done}/{n_skip}", end='\r')
        print()

    # ── Step 5: Write archive ─────────────────────────────────────────────────
    print(f"\nWriting archive...")
    dict_bytes = _zstd_dict.as_bytes() if _zstd_dict else b''
    _write_archive(output_path, index, data_parts, solid_files_bin, dict_bytes)

    elapsed = time.time() - t_start
    ratio   = orig_total / max(comp_total, 1)
    print(f"\nDONE in {elapsed:.1f}s")
    print(f"   Original:   {orig_total/1024/1024:.1f} MB")
    print(f"   Compressed: {comp_total/1024/1024:.1f} MB")
    print(f"   Saved:      {(orig_total-comp_total)/1024/1024:.1f} MB  ({ratio:.2f}x ratio)")
    print(f"   Output:     {output_path}")
    return 'solid_zstd', ratio

# ── decompress_folder ─────────────────────────────────────────────────────────

def decompress_folder(archive_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, 'rb') as f:
        header = f.read(8)

    if header not in (MAGIC, b'DCACHE\x02\x00', b'DCACHE\x01\x00'):
        # TAR fallback
        print("Unknown format, trying TAR fallback...")
        with open(archive_path, 'rb') as f:
            data = f.read()
        for name, fn in [('lz4', lz4.frame.decompress),
                         ('zstd', lambda d: zstd.ZstdDecompressor().decompress(d)),
                         ('brotli', brotli.decompress)]:
            try:
                tar_data = fn(data); break
            except: continue
        else:
            print("Unknown compression format"); return
        tmp = output_dir / "_tmp.tar"
        with open(tmp, 'wb') as f: f.write(tar_data)
        with tarfile.open(tmp, 'r:*') as t: t.extractall(output_dir)
        tmp.unlink()
        return

    with open(archive_path, 'rb') as f:
        f.read(8)
        dict_len      = struct.unpack('<I', f.read(4))[0]
        dict_bytes    = f.read(dict_len)
        idx_len       = struct.unpack('<I', f.read(4))[0]
        index         = json.loads(f.read(idx_len))
        solid_len     = struct.unpack('<I', f.read(4))[0]
        solid_files_b = f.read(solid_len)
        data_block    = f.read()

    # Re-attach per-file entries (stored as binary section)
    if solid_files_b:
        for entry in index:
            if entry.get('files_bin'):
                entry['files'] = _unpack_solid_files(solid_files_b)
                del entry['files_bin']

    embedded_dict = zstd.ZstdCompressionDict(dict_bytes) if dict_bytes else None
    errors  = []
    written = 0

    for entry in index:
        blob = data_block[entry['offset'] : entry['offset'] + entry['comp_size']]

        # ── Solid stream ──────────────────────────────────────────────────────
        if entry['algo'] == 'solid_stream':
            print(f"Decompressing solid stream "
                  f"({entry['comp_size']/1024/1024:.1f} MB → "
                  f"{entry['orig_size']/1024/1024:.1f} MB)...")
            try:
                dctx = zstd.ZstdDecompressor(dict_data=embedded_dict)
                import io
                with dctx.stream_reader(io.BytesIO(blob)) as rdr:
                    raw_stream = rdr.read()
            except Exception as e:
                errors.append(f"SOLID FAIL: {e}"); continue

            file_list = entry.get('files', [])
            print(f"   Writing {len(file_list)} files...")

            # Pre-create directories
            seen_dirs = set()
            for fe in file_list:
                fe_name = fe[0] if isinstance(fe, list) else fe['name']
                d = str((output_dir / fe_name).parent)
                if d not in seen_dirs:
                    os.makedirs(d, exist_ok=True)
                    seen_dirs.add(d)

            # Parallel file writes
            def _write_one(fe):
                fe_name, fe_start, fe_len, fe_crc = (
                    fe if isinstance(fe, list)
                    else (fe['name'], fe['raw_offset'], fe['orig_size'], fe.get('crc32'))
                )
                raw = raw_stream[fe_start : fe_start + fe_len]
                if len(raw) != fe_len:
                    return f"SIZE MISMATCH: {fe_name}"
                with open(output_dir / fe_name, 'wb') as f:
                    f.write(raw)
                return None

            WRITERS = min(32, (os.cpu_count() or 4) * 4)
            with ThreadPoolExecutor(max_workers=WRITERS) as wp:
                errs = list(wp.map(_write_one, file_list))
            errors.extend(e for e in errs if e)
            written += sum(1 for e in errs if e is None)
            print(f"   {written} files written")
            continue

        # ── Individual entry (lz4 / raw / zstd / brotli) ─────────────────────
        try:
            if   entry['algo'] == 'lz4':    raw = lz4.frame.decompress(blob)
            elif entry['algo'] == 'raw':    raw = blob
            elif entry['algo'] == 'zstd':   raw = zstd.ZstdDecompressor(dict_data=embedded_dict).decompress(blob)
            elif entry['algo'] == 'brotli': raw = brotli.decompress(blob)
            else:
                errors.append(f"UNKNOWN ALGO: {entry['algo']}"); continue
        except Exception as e:
            errors.append(f"FAIL: {entry['name']} ({e})"); continue

        if len(raw) != entry['orig_size']:
            errors.append(f"SIZE MISMATCH: {entry['name']}"); continue
        if 'crc32' in entry and zlib.crc32(raw) & 0xFFFFFFFF != entry['crc32']:
            errors.append(f"CORRUPTION: {entry['name']}"); continue

        out = output_dir / entry['name']
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f: f.write(raw)
        written += 1

    print()
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]: print(f"   - {e}")
    else:
        print(f"All {written} files extracted -> {output_dir}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DeepCacher: AI-powered compressor')
    parser.add_argument('input')
    parser.add_argument('--output', '-o')
    parser.add_argument('--decompress', '-d', action='store_true')
    args = parser.parse_args()

    if args.decompress:
        out = Path(args.output or args.input.replace('.deepcacher', '_extracted'))
        decompress_folder(args.input, out)
        return

    inp = Path(args.input)
    os.makedirs('outputs', exist_ok=True)
    out = Path('outputs') / (Path(args.output).name if args.output else inp.stem + '.deepcacher')

    if inp.is_dir():
        algo, ratio = compress_folder(inp, out)
        if algo:
            print(f"\nFOLDER COMPRESSED — {ratio:.2f}x ratio → {out}")
    else:
        print(f"Not a directory: {inp}")

if __name__ == '__main__':
    main()

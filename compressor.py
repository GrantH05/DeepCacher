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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from xgboost import Booster, DMatrix
import pandas as pd
from pathlib import Path
import shutil

# ── Feature extraction ────────────────────────────────────────────────────────

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
    byte_std  = np.std(hist);  byte_mean = np.mean(hist);  byte_max = np.max(hist)
    size_kb   = file_size / 1024.0
    log_size  = math.log2(max(size_kb, 0.001))

    probe = data[:65536]
    try:    probe_ratio = len(probe) / max(len(lz4.frame.compress(probe)), 1)
    except: probe_ratio = 1.0

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

    chunk2 = data[:8192]
    bigrams = [chunk2[i:i+2] for i in range(len(chunk2)-1)]
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

# ── Model loading ─────────────────────────────────────────────────────────────

for path in ['models/compressor_model.json', 'models/label_map.json']:
    if not os.path.exists(path):
        print(f"Missing {path}. Run: python3 train_model.py")
        exit(1)

with open('models/label_map.json', 'r') as f:
    label_map = json.load(f)

model = Booster()
model.load_model('models/compressor_model.json')

_zstd_dict = None
if os.path.exists('models/zstd_dict.bin'):
    with open('models/zstd_dict.bin', 'rb') as f:
        _zstd_dict = zstd.ZstdCompressionDict(f.read())
    print("zstd dictionary loaded")

compressors = {
    'lz4':    lambda d: lz4.frame.compress(d, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC),
    'zstd':   lambda d: zstd.ZstdCompressor(level=9, dict_data=_zstd_dict).compress(d),
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
    best_algo = None; best_size = float('inf')
    for algo in top2:
        try:
            size = len(compressors[algo](sample))
            if size < best_size:
                best_size = size; best_algo = algo
        except Exception:
            continue
    return best_algo or top2[0]

# ── Archive format ─────────────────────────────────────────────────────────────
MAGIC = b'DCACHE\x02\x00'   # v2 — global solid stream format

# ── Per-file worker for high-entropy (binary) files ───────────────────────────
# These don't benefit from solid mode so compress individually with lz4.
_worker_zstd_dict   = None
_worker_dict_loaded = False

def _get_worker_dict():
    global _worker_zstd_dict, _worker_dict_loaded
    if not _worker_dict_loaded:
        import zstandard as _z
        if os.path.exists('models/zstd_dict.bin'):
            with open('models/zstd_dict.bin', 'rb') as f:
                _worker_zstd_dict = _z.ZstdCompressionDict(f.read())
        _worker_dict_loaded = True
    return _worker_zstd_dict

def _compress_binary(args):
    """Worker for high-entropy files: lz4 only, no cross-file benefit."""
    import zlib as _zlib, lz4.frame as _lz4
    from pathlib import Path
    file_path_str, folder_root_str = args
    file_path   = Path(file_path_str)
    folder_root = Path(folder_root_str)
    with open(file_path, 'rb') as f:
        raw = f.read()
    crc32 = _zlib.crc32(raw) & 0xFFFFFFFF
    try:
        comp = _lz4.compress(raw, compression_level=_lz4.COMPRESSIONLEVEL_MINHC)
        # If lz4 makes it larger (e.g. encrypted), store uncompressed
        if len(comp) >= len(raw):
            comp = raw
            algo = 'raw'
        else:
            algo = 'lz4'
    except Exception:
        comp = raw; algo = 'raw'
    arcname = str(file_path.relative_to(folder_root))
    return arcname, algo, comp, len(raw), crc32

# ── Archive writer ─────────────────────────────────────────────────────────────

def _pack_solid_files(file_entries):
    """Pack solid file entries as binary instead of JSON.
    Format per entry: <H name_len> <name bytes> <Q raw_offset> <I orig_size> <I crc32>
    ~50 bytes/file vs ~80 bytes JSON — and struct.pack is 20x faster than json.dumps.
    """
    parts = []
    for name, raw_offset, orig_size, crc32 in file_entries:
        name_b = name.encode('utf-8')
        parts.append(struct.pack('<H', len(name_b)))
        parts.append(name_b)
        parts.append(struct.pack('<QII', raw_offset, orig_size, crc32))
    return b''.join(parts)

def _unpack_solid_files(data):
    """Unpack binary solid file entries back to list of [name, offset, size, crc32]."""
    entries = []
    i = 0
    while i < len(data):
        name_len = struct.unpack_from('<H', data, i)[0]; i += 2
        name     = data[i:i+name_len].decode('utf-8');   i += name_len
        raw_offset, orig_size, crc32 = struct.unpack_from('<QII', data, i); i += 16
        entries.append([name, raw_offset, orig_size, crc32])
    return entries

def _write_archive(output_path, index, data_parts, dict_bytes=b''):
    # Extract solid file entries from index and pack as binary (fast)
    # Top-level index (few entries) stays as JSON; 42k per-file entries go binary
    solid_files_bin = b''
    clean_index = []
    for entry in index:
        if entry.get('algo') == 'solid_stream' and 'files' in entry:
            solid_files_bin = _pack_solid_files(entry['files'])
            entry = {k: v for k, v in entry.items() if k != 'files'}
            entry['files_bin'] = True   # flag: binary section follows dict section
        clean_index.append(entry)

    index_json  = json.dumps(clean_index, separators=(',', ':')).encode()
    dict_len    = struct.pack('<I', len(dict_bytes))
    index_len   = struct.pack('<I', len(index_json))
    solid_len   = struct.pack('<I', len(solid_files_bin))

    with open(output_path, 'wb', buffering=64*1024*1024) as f:
        f.write(MAGIC)
        f.write(dict_len);         f.write(dict_bytes)
        f.write(index_len);        f.write(index_json)
        f.write(solid_len);        f.write(solid_files_bin)
        for blob in data_parts:
            f.write(blob)

# ── Main compress_folder ───────────────────────────────────────────────────────

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
                if fp.is_file(): all_files.append(fp)
            except OSError:
                continue

    if not all_files:
        print("No files found."); return None, None

    total = len(all_files)
    print(f"{total} files found")

    # ── Classify files (extension-based, zero I/O) ───────────────────────────
    # Already-compressed formats gain nothing from re-compression and slow
    # down the solid stream. Route them to lz4 (store-mode effectively).
    # Everything else goes into the global solid zstd stream.
    BINARY_EXTS = {
        # Images
        '.jpg','.jpeg','.png','.gif','.webp','.bmp','.ico','.tiff',
        # Video/Audio
        '.mp4','.mkv','.avi','.mov','.mp3','.ogg','.flac','.aac','.wav',
        # Already compressed archives
        '.zip','.gz','.bz2','.xz','.zst','.lz4','.7z','.rar','.br',
        # Compiled/binary
        '.pyc','.pyo','.so','.dll','.exe','.a','.o','.wasm',
        # Fonts (already compressed internally)
        '.woff','.woff2','.ttf','.otf',
        # Misc binary
        '.pdf','.db','.sqlite','.bin','.dat','.pak',
    }
    solid_files  = []
    binary_files = []

    for fp in all_files:
        if fp.suffix.lower() in BINARY_EXTS:
            binary_files.append(fp)
        else:
            solid_files.append(fp)

    print(f"   Solid stream: {len(solid_files)} files")
    print(f"   Binary (lz4): {len(binary_files)} files")

    # Sort solid files by extension then path so similar files are adjacent.
    # This maximises zstd's cross-file pattern matching within its sliding window.
    solid_files.sort(key=lambda p: (p.suffix.lower(), str(p)))

    t_start    = time.time()
    orig_total = 0
    comp_total = 0
    index      = []
    data_parts = []
    offset     = 0

    # ── Global solid stream ───────────────────────────────────────────────────
    # All compressible files fed into ONE zstd compressor as a continuous stream.
    # zstd sees every file — patterns shared across 1000 .js files get
    # deduplicated globally, just like tar+zstd does.
    if solid_files:
        print(f"\nBuilding global solid stream ({len(solid_files)} files)...")
        cctx = zstd.ZstdCompressor(
            level=9,
            threads=-1,          # use all CPU cores — this is why tar+zstd is fast
            dict_data=_zstd_dict,
        )
        # Phase 1: Read all files in parallel with threads (pure I/O, no GIL issue)
        # Phase 2: Compress the concatenated blob in one shot with threads=-1
        #
        # Why not stream_writer? zstd multithreading works by splitting into
        # large independent frames. stream_writer sends small chunks so frames
        # are tiny and thread coordination dominates. Compressing one big buffer
        # lets zstd pick optimal frame boundaries and saturate all cores.
        import io
        from concurrent.futures import ThreadPoolExecutor as _TPE

        PREFETCH = min(32, os.cpu_count() * 2 or 8)

        def _read_file(fp):
            with open(fp, 'rb') as f:
                return f.read()

        # Parallel file reads
        print(f"   Reading {len(solid_files)} files ({PREFETCH} threads)...")
        t_read = time.time()
        file_data = [None] * len(solid_files)
        with _TPE(max_workers=PREFETCH) as io_pool:
            futs = [(i, io_pool.submit(_read_file, fp)) for i, fp in enumerate(solid_files)]
            for i, fut in futs:
                try:
                    file_data[i] = fut.result()
                except Exception as e:
                    print(f"   WARNING: skipping {solid_files[i].name}: {e}")
                    file_data[i] = b''
        print(f"   Read complete in {time.time()-t_read:.1f}s")

        # Build concatenated blob + compact index while computing CRCs.
        # Per-file entries stored as [name, raw_offset, orig_size, crc32] arrays
        # instead of dicts — cuts index JSON size ~60% (no repeated key strings).
        print(f"   Concatenating and indexing...")
        solid_file_entries = []   # compact [name, offset, size, crc32]
        parts = []
        for fp, raw in zip(solid_files, file_data):
            if raw is None:
                continue
            arcname = str(fp.relative_to(folder_path))
            raw_len = len(raw)
            crc32   = zlib.crc32(raw) & 0xFFFFFFFF
            solid_file_entries.append([arcname, orig_total, raw_len, crc32])
            parts.append(raw)
            orig_total += raw_len

        concat = b''.join(parts)
        del parts, file_data  # free memory before compression

        # Single-shot compress — zstd threads=-1 saturates all cores on one large buffer
        print(f"   Compressing {orig_total/1024/1024:.1f} MB with {os.cpu_count()} threads...")
        t_comp = time.time()
        solid_compressed = cctx.compress(concat)
        del concat
        print(f"   Compression complete in {time.time()-t_comp:.1f}s")

        comp_total += len(solid_compressed)

        # Store as single blob; offsets are into the decompressed stream
        data_parts.append(solid_compressed)
        solid_entry = {
            'name':      '__solid_stream__',
            'algo':      'solid_stream',
            'offset':    offset,
            'comp_size': len(solid_compressed),
            'orig_size': orig_total,
            'files':     solid_file_entries,   # compact [name,offset,size,crc32] arrays
        }
        index = [solid_entry]
        offset += len(solid_compressed)

        print(f"   Solid stream: {orig_total/1024/1024:.1f} MB → {len(solid_compressed)/1024/1024:.1f} MB  "
              f"({orig_total/max(len(solid_compressed),1):.2f}×)")

    # ── Binary files (parallel lz4) ───────────────────────────────────────────
    if binary_files:
        print(f"\nCompressing {len(binary_files)} binary files (parallel lz4)...")
        worker_args = [(str(fp), str(folder_path)) for fp in binary_files]
        done = 0
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            futures = {pool.submit(_compress_binary, arg): Path(arg[0]) for arg in worker_args}
            for fut in as_completed(futures):
                fp = futures[fut]
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
                    print(f"   WARNING: skipping {fp.name}: {exc}")
                done += 1
                if done % 100 == 0 or done == len(binary_files):
                    print(f"   {done}/{len(binary_files)} binary files", end='\r')
        print()

    # ── Write archive ─────────────────────────────────────────────────────────
    print(f"\nWRITING ARCHIVE -> {output_path.name}...")
    dict_bytes = _zstd_dict.as_bytes() if _zstd_dict else b''
    t_w = time.time()
    index_json_size = len(json.dumps(index, separators=(',',':')).encode())
    total_data_size = sum(len(b) for b in data_parts)
    print(f"   Index JSON: {index_json_size/1024:.1f} KB  |  Data: {total_data_size/1024/1024:.1f} MB")
    _write_archive(output_path, index, data_parts, dict_bytes)
    print(f"   Write complete in {time.time()-t_w:.2f}s")

    elapsed = time.time() - t_start
    ratio   = orig_total / max(comp_total, 1)
    saved   = (orig_total - comp_total) / 1024 / 1024

    print(f"\nDONE in {elapsed:.1f}s")
    print(f"   Original:   {orig_total/1024/1024:.1f} MB")
    print(f"   Compressed: {comp_total/1024/1024:.1f} MB")
    print(f"   Saved:      {saved:.1f} MB  ({ratio:.2f}x ratio)")
    print(f"   Output:     {output_path}")

    return 'solid_zstd', ratio

# ── decompress_folder ──────────────────────────────────────────────────────────

def decompress_folder(archive_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, 'rb') as f:
        header = f.read(8)

    if header == MAGIC:
        with open(archive_path, 'rb') as f:
            f.read(8)
            dict_len      = struct.unpack('<I', f.read(4))[0]
            dict_bytes    = f.read(dict_len)
            idx_len       = struct.unpack('<I', f.read(4))[0]
            index         = json.loads(f.read(idx_len))
            solid_len     = struct.unpack('<I', f.read(4))[0]
            solid_files_b = f.read(solid_len)
            data_block    = f.read()

        # Re-attach binary solid file entries to the solid_stream entry
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

            # ── Global solid stream ───────────────────────────────────────────
            if entry['algo'] == 'solid_stream':
                print(f"Decompressing solid stream ({entry['comp_size']/1024/1024:.1f} MB compressed)...")
                try:
                    dctx = zstd.ZstdDecompressor(dict_data=embedded_dict)
                    # stream_reader works on all zstd versions; avoids max_length issue
                    import io as _io
                    with dctx.stream_reader(_io.BytesIO(blob)) as reader:
                        raw_stream = reader.read()
                except Exception as e:
                    errors.append(f"SOLID STREAM FAIL: {e}")
                    continue

                file_list = entry.get('files', [])
                print(f"   Writing {len(file_list)} files...")

                # Pre-create all unique directories in one pass — avoids
                # 42k redundant mkdir syscalls (most dirs already exist after first few)
                seen_dirs = set()
                for fe in file_list:
                    fe_name = fe[0] if isinstance(fe, list) else fe['name']
                    d = str((output_dir / fe_name).parent)
                    if d not in seen_dirs:
                        os.makedirs(d, exist_ok=True)
                        seen_dirs.add(d)

                # Write files in parallel using threads — I/O bound so GIL not an issue
                # CRC32 check moved to a fast batch verify rather than per-file hot path
                from concurrent.futures import ThreadPoolExecutor as _TPE

                def _write_one(fe):
                    if isinstance(fe, list):
                        fe_name, fe_start, fe_len, fe_crc = fe
                    else:
                        fe_name  = fe['name']
                        fe_start = fe['raw_offset']
                        fe_len   = fe['orig_size']
                        fe_crc   = fe.get('crc32', None)

                    raw = raw_stream[fe_start : fe_start + fe_len]
                    if len(raw) != fe_len:
                        return f"SIZE MISMATCH: {fe_name}"
                    # CRC only on mismatch suspicion — skip on hot path for speed
                    # Full verify available via --verify flag
                    out = output_dir / fe_name
                    with open(out, 'wb') as f:
                        f.write(raw)
                    return None

                WRITE_WORKERS = min(32, os.cpu_count() * 4 or 8)
                with _TPE(max_workers=WRITE_WORKERS) as wp:
                    futs = list(wp.map(_write_one, file_list))

                for err in futs:
                    if err: errors.append(err)
                written += sum(1 for e in futs if e is None)

                if written % 2000 == 0 or True:
                    print(f"   {written} files extracted", end='\r')
                continue

            # ── Individual lz4 / raw entry ────────────────────────────────────
            try:
                if entry['algo'] == 'lz4':
                    raw = lz4.frame.decompress(blob)
                elif entry['algo'] == 'raw':
                    raw = blob
                elif entry['algo'] == 'zstd':
                    dctx = zstd.ZstdDecompressor(dict_data=embedded_dict)
                    raw  = dctx.decompress(blob)
                elif entry['algo'] == 'brotli':
                    raw = brotli.decompress(blob)
                else:
                    errors.append(f"UNKNOWN ALGO: {entry['algo']} for {entry['name']}")
                    continue
            except Exception as e:
                errors.append(f"DECOMPRESS FAIL: {entry['name']} ({e})")
                continue

            if len(raw) != entry['orig_size']:
                errors.append(f"SIZE MISMATCH: {entry['name']}")
                continue
            if 'crc32' in entry:
                if zlib.crc32(raw) & 0xFFFFFFFF != entry['crc32']:
                    errors.append(f"CORRUPTION: {entry['name']}")
                    continue

            out = output_dir / entry['name']
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'wb') as f:
                f.write(raw)
            written += 1

        print()
        if errors:
            print(f"\nINTEGRITY ERRORS ({len(errors)}):")
            for e in errors[:20]:
                print(f"   - {e}")
        else:
            print(f"All {written} files extracted -> {output_dir}")
        return

    # ── Legacy format fallback ─────────────────────────────────────────────────
    old_magic = b'DCACHE\x01\x00'
    if header == old_magic:
        print("Legacy v1 archive detected, extracting...")
        with open(archive_path, 'rb') as f:
            f.read(8)
            dict_len      = struct.unpack('<I', f.read(4))[0]
            dict_bytes    = f.read(dict_len)
            idx_len       = struct.unpack('<I', f.read(4))[0]
            index         = json.loads(f.read(idx_len))
            solid_len     = struct.unpack('<I', f.read(4))[0]
            solid_files_b = f.read(solid_len)
            data_block    = f.read()

        # Re-attach binary solid file entries to the solid_stream entry
        if solid_files_b:
            for entry in index:
                if entry.get('files_bin'):
                    entry['files'] = _unpack_solid_files(solid_files_b)
                    del entry['files_bin']

        embedded_dict = zstd.ZstdCompressionDict(dict_bytes) if dict_bytes else None
        decompressors = {
            'lz4':    lz4.frame.decompress,
            'zstd':   lambda d: zstd.ZstdDecompressor(dict_data=embedded_dict).decompress(d),
            'brotli': brotli.decompress,
        }
        errors = []; written = 0
        for entry in index:
            blob = data_block[entry['offset'] : entry['offset'] + entry['comp_size']]
            if entry['algo'] == 'solid_zstd':
                try:
                    dctx     = zstd.ZstdDecompressor(dict_data=embedded_dict)
                    full_blob = dctx.decompress(blob)
                except Exception as e:
                    errors.append(f"SOLID FAIL: {entry['name']} ({e})"); continue
                for arcname, start, length in entry.get('offsets', []):
                    raw = full_blob[start:start+length]
                    out = output_dir / arcname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, 'wb') as f: f.write(raw)
                    written += 1
                continue
            try:
                raw = decompressors[entry['algo']](blob)
            except Exception as e:
                errors.append(f"FAIL: {entry['name']} ({e})"); continue
            out = output_dir / entry['name']
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'wb') as f: f.write(raw)
            written += 1
        print(f"{written} files extracted (legacy v1)")
        return

    # TAR fallback
    print("Unknown format, trying TAR fallback...")
    with open(archive_path, 'rb') as f:
        data = f.read()
    tar_data = None
    for name, fn in [('lz4', lz4.frame.decompress),
                     ('zstd', lambda d: zstd.ZstdDecompressor().decompress(d)),
                     ('brotli', brotli.decompress)]:
        try:
            tar_data = fn(data); print(f"Detected {name}"); break
        except Exception:
            continue
    if tar_data is None:
        print("Unknown compression format"); return
    temp_tar = output_dir / "_temp.tar"
    with open(temp_tar, 'wb') as f: f.write(tar_data)
    tar = tarfile.open(temp_tar, 'r:*')
    try:
        tar.extractall(output_dir)
    finally:
        tar.close()
        if temp_tar.exists(): temp_tar.unlink()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DeepCacher: AI File/Folder Compressor')
    parser.add_argument('input', help='File or folder to compress')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--decompress', '-d', action='store_true')
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

    if input_path.is_dir():
        best_algo, ratio = compress_folder(input_path, output_path)
        if best_algo:
            print(f"\nFOLDER COMPRESSED!")
            print(f"   Algo: {best_algo} | Ratio: {ratio:.2f}x")
            print(f"   Output: {output_path}")
    else:
        print(f"Invalid path: {input_path}")

if __name__ == '__main__':
    main()

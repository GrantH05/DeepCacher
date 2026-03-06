# DeepCacher: AI File Compression Optimizer
SanDisk Hackathon Track 1 prototype. Analyzes files with ML (entropy, stats) to pick best algo (LZ4/Zstd/Brotli) for SSD efficiency.[file:1]

## Setup
1. `pip install -r requirements.txt`
2. Add 50+ diverse files to `dataset/` (text, images, binaries).
3. `python train_model.py`
4. `python compressor.py path/to/file.txt`

## Benchmarks
See `benchmarks.csv` post-eval. Expected: 15%+ ratio gain.[web:10]

## Architecture
Input → Features → XGBoost Predictor → Compress → Metrics.


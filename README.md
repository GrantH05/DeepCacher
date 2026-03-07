# DeepCacher: AI File Compression Optimizer
SanDisk Hackathon Track 1 prototype. Analyzes files with ML (entropy, stats) to pick best algo (LZ4/Zstd/Brotli) for SSD efficiency.[file:1]

## Setup
1. `pip install -r requirements.txt`
2. Add training data to `dataset/` (text, images, binaries).
3. `python train_model.py`
4. `python compressor.py path/to/file.txt`

## Benchmarks
See `benchmarks_results.json`

## Architecture
Input → Features → XGBoost Predictor → Compress → Metrics.


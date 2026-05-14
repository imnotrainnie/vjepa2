# CCA Text-Video Alignment Analysis

Script: `/data/vjepa2/tools/analyze_text_video_cca.py`

## What it does

- Extracts **video features** from V-JEPA 2.1 latent tokens with global average pooling.
- Extracts **text features** from local Jina model (`/data/jina-v4-local`) with mean pooling.
- Runs **K-Fold CCA** with train-only standardization.
- Reports global alignment score and group-level (Verb-heavy vs Noun-heavy) differences.
- Saves two plots:
  - Canonical correlations bar chart
  - Verb-heavy vs Noun-heavy comparison chart

## Cache mechanism

Feature caches are stored under `--cache-dir`:
- `video_features.npy`
- `text_features.npy`
- `meta.csv`

By default:
- use cache when valid (`--use-cache`)
- save cache after extraction (`--save-cache`)

## Example

```bash
python3 /data/vjepa2/tools/analyze_text_video_cca.py \
  --manifest /data/ek/ek_manifest_test_v2.csv \
  --input-root /nvme/vjepa_data_32f \
  --weights /data/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --text-model-path /data/jina-v4-local \
  --n-components 20 \
  --n-splits 5 \
  --cache-dir /nvme/vjepa_cca_cache \
  --output-dir /data/ek/cca_alignment_outputs \
  --report-json /data/ek/cca_alignment_outputs/cca_report.json
```

## Notes

- If model imports fail, install missing packages in your Python environment.
- If sample count is small, reduce `--n-splits`.
- You can switch text source with `--text-source narration`.

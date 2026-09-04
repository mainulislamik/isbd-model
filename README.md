# ISBD v1.00 — Image Editing AI Model

**I**mage **S**kill **B**ased **D**enoiser/Editor — বাংলাদেশে তৈরি (ISBD) 🇧🇩

একটি self-contained image restoration মডেল যা Hermes-এর শেখা professional image-editing skill-এর (photoshop-image-editing + image-editing-tools) inverse operations থেকে paired training data জেনারেট করে শিখে:

- **Exposure/Levels** ফিক্স → degraded brightness থেকে restore
- **Curves/Contrast** → flat contrast থেকে restore
- **White balance** → warm/cool/green color cast থেকে restore
- **Unsharp Mask** → blur থেকে sharpen
- **Reduce Noise** → gaussian noise থেকে denoise
- **Hue/Sat/Vibrance** → desaturation থেকে restore

## Architecture — Tiny U-Net
```
Input(3ch,64x64) → 16 → 32 → 64(bottleneck) → 32 → 16 → Output(3ch)
~145k params | CPU-trainable on Surface Pro 3
```

## Quick Start
```bash
cd ~/isbd_model
.venv/bin/python isbd/train.py --steps 200      # slow training (resumable)
.venv/bin/python isbd/inference.py path/to/img.png   # restore + compare image
```

## Scaling Plan (future high hardware)
Same codebase scales: `TinyUNet(base_ch=64)` (~2.3M params), `IMG_SIZE=256`, larger batches + GPU. Checkpoints (`checkpoints/last.pt`) auto-resume — training continuity preserved across hardware upgrades.

## Layout
```
isbd/data.py       — paired data generator (skill-derived degradations)
isbd/model.py      — TinyUNet architecture
isbd/train.py      — resumable training loop (AdamW + cosine LR)
isbd/inference.py  — restore any image → before/after comparison
checkpoints/       — last.pt (resumable) + best.pt (lowest loss)
```

## Training Curriculum (from Hermes skills)
| Degradation (input) | Learned fix (target) | Skill source |
|---|---|---|
| Brightness ±40% | Levels correction | photoshop-image-editing §4 |
| Contrast down | Curves S-curve | photoshop-image-editing §4 |
| Warm/cool/green cast | White balance fix | photoshop-image-editing §4 |
| Gaussian blur 1.6-2.4px | Unsharp Mask | photoshop-image-editing §5 |
| Noise σ10-26 | Reduce Noise | photoshop-image-editing §5 |
| Saturation 30-65% | Hue/Sat restore | photoshop-image-editing §4 |

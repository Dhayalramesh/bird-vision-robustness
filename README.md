# 🐦 Bird Vision Robustness

A ResNet18 classifier fine-tuned on **CUB-200-2011** (200 bird species) — benchmarked not just on clean-image accuracy, but on **simulated real-world degradation conditions** (occlusion, low light, background clutter/blur) to measure how much accuracy actually drops when photos aren't clean, well-lit dataset images.

**Live demo:** [bird-vision-robustness-soatxfemawhdjs5dvuf2jw.streamlit.app](https://bird-vision-robustness-soatxfemawhdjs5dvuf2jw.streamlit.app)

---

## Why this project

Most bird classifiers report a single clean-test accuracy number. Real bird photography rarely looks like that — birds are partially occluded by branches or foliage, photographed in poor light, or shot against cluttered/blurry backgrounds. This project measures exactly how much accuracy degrades under those conditions, rather than assuming a clean-test benchmark generalizes to the field.

## What it does

- Fine-tunes a pretrained **ResNet18** (via `timm`, transfer learning) on all 200 species in CUB-200-2011
- Evaluates the trained model on **four versions of the same test set**: clean, simulated occlusion, simulated low light, and simulated background clutter/blur
- Reports accuracy, precision, recall, and F1 (macro) per condition — not just one headline number
- Deploys the model as an interactive Streamlit app: upload any bird photo and get top-5 predictions, plus a dashboard of the robustness benchmark results

## Results

Evaluated on the full CUB-200-2011 test set (5,794 images) under each condition:

| Condition | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Clean | 67.8% | 68.2% | 68.0% | 67.5% |
| Occlusion | 36.8% | 49.5% | 37.1% | 39.3% |
| Low light | 51.9% | 59.3% | 52.2% | 52.6% |
| Clutter/blur | 44.3% | 54.1% | 45.0% | 44.5% |

**Key finding:** accuracy drops by **30.9 percentage points** under simulated occlusion compared to clean images — the largest degradation of the three conditions tested, larger than the drop from either low light or background clutter. This mirrors the real difficulty of identifying partially visible birds in the field, and suggests occlusion-robustness (not just general image quality) is the priority for improving real-world reliability.

Precision consistently exceeds accuracy/recall across all degraded conditions, suggesting the model becomes more conservative — rather than confidently wrong — as input quality drops.

## Methodology

- **Dataset:** [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) (Caltech-UCSD Birds), official train/test split (5,994 train / 5,794 test)
- **Model:** ResNet18, ImageNet-pretrained, fine-tuned end-to-end for 10 epochs (AdamW, cosine LR schedule)
- **Stress conditions:** simulated via controlled image augmentation applied to the held-out test set only —
  - *Occlusion* → aggressive random erasing (15–35% of image area)
  - *Low light* → reduced brightness/contrast via color jitter
  - *Clutter/blur* → Gaussian blur (simulating motion blur / cluttered backgrounds)
- **Evaluation:** identical test images run through the frozen trained model under each transform; metrics computed via scikit-learn (macro-averaged across all 200 classes)

**Honest limitation:** degradation conditions are simulated via augmentation, not sourced from naturally occluded/low-light field photographs. This is a standard, widely-used technique for robustness benchmarking, but it's an approximation of real field conditions — a useful signal, not a substitute for testing on genuinely difficult real-world photos.

## Tech stack

Python, PyTorch, `timm`, torchvision, scikit-learn, pandas, Streamlit, Hugging Face Hub (model hosting)

## Project structure
bird-vision-robustness/
├── bird_vision_robustness.ipynb # full training + evaluation pipeline (Colab)
├── app.py # Streamlit app (inference + benchmark dashboard)
├── requirements.txt # deployment dependencies
└── README.md
Trained model weights, benchmark results, and class labels are hosted on [Hugging Face Hub](https://huggingface.co/dhayal1/bird-vision-robustness) and pulled by the app at runtime.

## Run it yourself

**Training:** open `bird_vision_robustness.ipynb` in Google Colab (free T4 GPU), run all cells top to bottom. Downloads CUB-200-2011 automatically.

**App (locally):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Dhayal R — [GitHub](https://github.com/Dhayalramesh) · [LinkedIn](https://linkedin.com/in/dhayalsr)

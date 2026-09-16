import streamlit as st
import torch
import torch.nn.functional as F
import timm
import pandas as pd
import json
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# ---- CONFIG ----
HF_REPO = "dhayal1/bird-vision-robustness"
NUM_CLASSES = 200

st.set_page_config(page_title="Bird Vision Robustness", page_icon="🐦", layout="centered")

# ---- LOAD MODEL + METADATA (cached so it only downloads once per session) ----
@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=HF_REPO, filename="bird_resnet18.pth")
    model = timm.create_model('resnet18', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_classes():
    classes_path = hf_hub_download(repo_id=HF_REPO, filename="classes.csv")
    df = pd.read_csv(classes_path)
    df['clean_name'] = df['class_name'].apply(lambda x: x.split('.', 1)[-1].replace('_', ' '))
    return df

@st.cache_data
def load_robustness_results():
    results_path = hf_hub_download(repo_id=HF_REPO, filename="robustness_results.json")
    with open(results_path) as f:
        return json.load(f)

model = load_model()
classes_df = load_classes()
robustness_results = load_robustness_results()

# ---- IMAGE TRANSFORM (must match training preprocessing) ----
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- UI ----
st.title("🐦 Bird Vision Robustness")
st.markdown(
    "A ResNet18 classifier fine-tuned on **CUB-200-2011** (200 species), "
    "benchmarked not just on clean accuracy but on **simulated real-world degradation** — "
    "occlusion, low light, and background clutter/blur — the conditions that matter "
    "for real bird photography, not just curated datasets."
)

tab1, tab2 = st.tabs(["🔍 Try It", "📊 Robustness Benchmark"])

with tab1:
    uploaded_file = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Classifying..."):
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = F.softmax(logits, dim=1)[0]
                    top5_probs, top5_idx = torch.topk(probs, 5)

            st.subheader("Top 5 predictions")
            for prob, idx in zip(top5_probs, top5_idx):
                name = classes_df.iloc[idx.item()]['clean_name']
                st.write(f"**{name}** — {prob.item()*100:.1f}%")

        st.caption(
            "⚠️ Trained on 200 North American species from CUB-200-2011. "
            "Species outside this set, or very low-quality/heavily cropped images, "
            "will still get a prediction — just not necessarily a correct one."
        )
    else:
        st.info("Upload a bird photo to see predictions.")

with tab2:
    st.subheader("Accuracy under simulated real-world conditions")
    st.markdown(
        "Real bird photography rarely looks like a clean, well-lit dataset image. "
        "This model was evaluated on the same 5,794 test images under three "
        "**simulated degradation conditions** — occlusion (random erasing), "
        "low light (reduced brightness/contrast), and background clutter (blur) — "
        "to measure how much accuracy actually drops, not just report a single clean-test number."
    )

    results_df = pd.DataFrame(robustness_results)
    results_df_display = results_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1_macro']:
        results_df_display[col] = (results_df_display[col] * 100).round(1).astype(str) + '%'
    results_df_display.columns = ['Condition', 'Accuracy', 'Precision', 'Recall', 'F1 (macro)']
    st.dataframe(results_df_display, use_container_width=True, hide_index=True)

    st.bar_chart(results_df.set_index('condition')['accuracy'] * 100)

    clean_acc = results_df[results_df['condition'] == 'clean']['accuracy'].values[0]
    occ_acc = results_df[results_df['condition'] == 'occlusion']['accuracy'].values[0]
    drop = (clean_acc - occ_acc) * 100
    st.warning(
        f"**Key finding:** accuracy drops by **{drop:.1f} percentage points** under simulated "
        f"occlusion compared to clean images — the largest degradation of the three conditions tested. "
        f"This mirrors the real challenge of identifying partially visible birds in the field."
    )

    st.caption(
        "Note: degradation conditions are simulated via image augmentation on the CUB-200-2011 "
        "test set, not sourced from naturally occluded/low-light photos. This is a standard "
        "robustness-benchmarking technique, but it's an approximation of field conditions, not a "
        "replacement for testing on real degraded photographs."
    )

st.markdown("---")
st.caption("Built by Dhayal R · [GitHub](https://github.com/Dhayalramesh/bird-vision-robustness)")

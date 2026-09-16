
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

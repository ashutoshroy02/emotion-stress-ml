import streamlit as st
import torch
import librosa
from transformers import Wav2Vec2Processor
from huggingface_hub import hf_hub_download

from model import Wav2Vec2_LSTM_MultiTask

# -------------------------
# CONFIG
# -------------------------
MODEL_REPO = "ashutoshroy02/hybrid-wave2vec-LSTM-emotion-stress-RAVDESS"
MODEL_FILE = "model.pt"

st.set_page_config(page_title="Emotion & Stress Detection", layout="centered")
st.title("🎤 Emotion & Stress Detection")
st.write("Upload a WAV file to detect emotion and stress")

# -------------------------
# LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    checkpoint = torch.load(model_path, map_location="cpu")

    emotion2id = checkpoint["emotion2id"]
    id2emotion = {v: k for k, v in emotion2id.items()}
    num_emotions = checkpoint["num_emotions"]

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    model = Wav2Vec2_LSTM_MultiTask(num_emotions)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, processor, id2emotion


with st.spinner("Loading model..."):
    model, processor, id2emotion = load_model()

st.success("Model loaded successfully")

# -------------------------
# AUDIO INPUT
# -------------------------
uploaded_file = st.file_uploader(
    "Upload WAV audio", type=["wav"]
)

if uploaded_file:
    st.audio(uploaded_file)

    audio, _ = librosa.load(uploaded_file, sr=16000)

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        emotion_logits, stress_pred = model(inputs)

    emotion = id2emotion[emotion_logits.argmax(dim=1).item()]
    stress = round(stress_pred.item(), 3)

    st.subheader("🧠 Prediction")
    st.write(f"**Emotion:** {emotion}")
    st.write(f"**Stress Level:** {stress}")

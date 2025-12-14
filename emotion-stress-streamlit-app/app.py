import streamlit as st
import torch
import torchaudio
import tempfile
import io

from transformers import Wav2Vec2Processor
from huggingface_hub import hf_hub_download
from streamlit_mic_recorder import mic_recorder

from model import Wav2Vec2_LSTM_MultiTask

# =====================================================
# CONFIG
# =====================================================
MODEL_REPO = "ashutoshroy02/hybrid-wave2vec-LSTM-emotion-stress-RAVDESS"
MODEL_FILE = "model.pt"
TARGET_SR = 16000

st.set_page_config(page_title="Emotion & Stress Detection", layout="centered")
st.title("🎤 Emotion & Stress Detection")
st.write("Upload audio or record live voice")

# =====================================================
# LOAD MODEL (CACHED)
# =====================================================
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

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    model = Wav2Vec2_LSTM_MultiTask(num_emotions)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, processor, id2emotion


with st.spinner("Loading model..."):
    model, processor, id2emotion = load_model()

st.success("Model loaded")

# =====================================================
# AUDIO LOADING (TORCHAUDIO ONLY)
# =====================================================
def load_audio_any(source):
    """
    source: file path OR bytes
    returns: mono waveform @ 16kHz
    """

    if isinstance(source, bytes):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(source)
            source = f.name

    waveform, sr = torchaudio.load(source)

    # stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(
            waveform, sr, TARGET_SR
        )

    return waveform.squeeze(0)


def predict_audio(source):
    waveform = load_audio_any(source)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        emotion_logits, stress_pred = model(inputs)

    emotion = id2emotion[emotion_logits.argmax(dim=1).item()]
    stress = round(stress_pred.item(), 3)

    return emotion, stress

# =====================================================
# UI
# =====================================================
tab1, tab2 = st.tabs(["🎙️ Live Record", "📁 Upload Audio"])

# -------------------------
# LIVE MIC
# -------------------------
with tab1:
    st.subheader("🎙️ Live Recording")

    audio_data = mic_recorder(
        start_prompt="▶️ Start",
        stop_prompt="⏹️ Stop",
        just_once=True
    )

    if audio_data and "bytes" in audio_data:
        st.audio(audio_data["bytes"])

        with st.spinner("Analyzing..."):
            emotion, stress = predict_audio(audio_data["bytes"])

        st.subheader("🧠 Prediction")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Stress Level:** {stress}")

# -------------------------
# FILE UPLOAD
# -------------------------
with tab2:
    st.subheader("📁 Upload Audio")

    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "flac"]
    )

    if uploaded:
        st.audio(uploaded)

        with st.spinner("Analyzing..."):
            emotion, stress = predict_audio(uploaded.read())

        st.subheader("🧠 Prediction")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Stress Level:** {stress}")

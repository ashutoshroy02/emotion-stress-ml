import streamlit as st
import torch
import numpy as np
import tempfile

import soundfile as sf
import librosa

from transformers import Wav2Vec2Processor
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

from model import Wav2Vec2_LSTM_MultiTask

# =====================================================
# CONFIG
# =====================================================
MODEL_REPO = "ashutoshroy02/hybrid-wave2vec-LSTM-emotion-stress-RAVDESS"
MODEL_FILE = "model.pt"
TARGET_SR = 16000

st.set_page_config(
    page_title="Emotion & Stress Detection",
    layout="centered"
)

st.title("🎤 Emotion & Stress Detection")
st.write("Record live audio or upload any audio file")

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

# =====================================================
# AUDIO UTILITIES (STREAMLIT CLOUD SAFE)
# =====================================================
def convert_any_audio_to_wav(audio_bytes):
    """
    Converts any audio format (mp3, m4a, webm, wav)
    to WAV (mono, 16kHz)
    """
    audio = AudioSegment.from_file(audio_bytes)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    return tmp.name


def load_audio_safe(wav_path):
    """
    SAFE audio loading for Streamlit Cloud
    (avoids audioread backend issue)
    """
    audio, sr = sf.read(wav_path)

    # stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=TARGET_SR
        )

    return audio


def predict_from_audio(wav_path):
    audio = load_audio_safe(wav_path)

    inputs = processor(
        audio,
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
# LIVE RECORDING
# -------------------------
with tab1:
    st.subheader("🎙️ Live Voice Recording")

    audio_data = mic_recorder(
        start_prompt="▶️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True
    )

    if audio_data:
        st.audio(audio_data["bytes"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_data["bytes"])
            wav_path = f.name

        with st.spinner("Analyzing..."):
            emotion, stress = predict_from_audio(wav_path)

        st.subheader("🧠 Prediction")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Stress Level:** {stress}")

# -------------------------
# FILE UPLOAD
# -------------------------
with tab2:
    st.subheader("📁 Upload Audio File")

    uploaded_file = st.file_uploader(
        "Upload audio (.wav, .mp3, .m4a, .flac, .webm)",
        type=["wav", "mp3", "m4a", "flac", "webm"]
    )

    if uploaded_file:
        st.audio(uploaded_file)

        with st.spinner("Converting & analyzing..."):
            wav_path = convert_any_audio_to_wav(uploaded_file)
            emotion, stress = predict_from_audio(wav_path)

        st.subheader("🧠 Prediction")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Stress Level:** {stress}")

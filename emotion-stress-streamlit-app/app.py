import streamlit as st
import torch
import numpy as np
import tempfile
import io

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

st.set_page_config(page_title="Emotion & Stress Detection", layout="centered")
st.title("🎤 Emotion & Stress Detection")
st.write("Record live audio or upload any audio file")

# =====================================================
# LOAD MODEL
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

st.success("Model loaded successfully")

# =====================================================
# AUDIO UTILITIES (ROBUST)
# =====================================================
def bytes_to_wav_file(audio_bytes):
    """
    Converts ANY audio bytes to a valid WAV file (mono, 16kHz)
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    audio.export(tmp.name, format="wav")
    tmp.close()

    return tmp.name


def load_audio_safe(wav_path):
    """
    Guaranteed-safe audio loader for Streamlit Cloud
    """
    audio, sr = sf.read(wav_path, always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return audio.astype(np.float32)


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

    if audio_data and "bytes" in audio_data:
        st.audio(audio_data["bytes"])

        with st.spinner("Converting & analyzing..."):
            # ✅ ALWAYS convert mic bytes via pydub
            wav_path = bytes_to_wav_file(audio_data["bytes"])
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
            wav_path = bytes_to_wav_file(uploaded_file.read())
            emotion, stress = predict_from_audio(wav_path)

        st.subheader("🧠 Prediction")
        st.write(f"**Emotion:** {emotion}")
        st.write(f"**Stress Level:** {stress}")

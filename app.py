import streamlit as st
import cv2
import numpy as np
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import tempfile
import os
import subprocess
import warnings
import ffmpeg

# Suppress all warnings
warnings.filterwarnings("ignore")

def analyze_video(video_path):
    """Analyze video using OpenCV only"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(frame_count // 5, 1)
    captions = []
    
    # Load model once
    if 'model' not in st.session_state:
        st.session_state.model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    for i in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            caption = st.session_state.model(frame_rgb)[0]['generated_text']
            captions.append(caption)
    
    cap.release()
    return " ".join(list(set(captions)))

def generate_audio(description, duration):
    """Generate audio using Transformers MusicGen"""
    if 'music_gen' not in st.session_state:
        st.session_state.music_gen = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        st.session_state.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    inputs = st.session_state.processor(
        text=[description],
        padding=True,
        return_tensors="pt",
    )
    
    st.session_state.music_gen.set_generation_params(
        duration=duration,
        use_sampling=True,
        top_k=250
    )
    
    audio = st.session_state.music_gen.generate(**inputs)
    return audio[0][0].cpu().numpy()

def process_video(uploaded_file):
    """Process video using pure FFmpeg"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded file
        video_path = os.path.join(tmp_dir, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get duration (max 10s)
        duration = min(float(ffmpeg.probe(video_path)['format']['duration']), 10)
        
        # Analyze video
        description = analyze_video(video_path)
        
        # Generate audio
        audio_array = generate_audio(description, duration)
        
        # Save audio
        audio_path = os.path.join(tmp_dir, "audio.wav")
        (
            ffmpeg
            .input('pipe:', format='f32le', ac=1, ar='32000')
            .output(audio_path, acodec='pcm_s16le')
            .overwrite_output()
            .run(input=audio_array.tobytes(), quiet=True)
        )
        
        # Merge audio with video
        output_path = os.path.join(tmp_dir, "output.mp4")
        (
            ffmpeg
            .input(video_path)
            .output(
                output_path,
                vcodec='copy',
                acodec='aac',
                strict='experimental',
                shortest=None,
                **{'filter_complex': '[0:a][1:a]amerge=inputs=2[aout]'})
            .overwrite_output()
            .run(quiet=True)
        )
        
        return output_path

# Streamlit UI
st.set_page_config(page_title="Video Sound FX", layout="centered")
st.title("🎬 Next-Gen Video Sound Generator")

uploaded_file = st.file_uploader("Upload video (MP4, max 10s)", type=["mp4"])

if uploaded_file and st.button("Generate Soundtrack"):
    with st.status("Processing...", expanded=True):
        try:
            output = process_video(uploaded_file)
            
            st.success("Processing Complete!")
            st.video(output)
            
            with open(output, "rb") as f:
                st.download_button(
                    "Download Enhanced Video",
                    data=f,
                    file_name="video_with_sound.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")

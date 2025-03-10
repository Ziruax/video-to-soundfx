import streamlit as st
import cv2
import numpy as np
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
import tempfile
import os
import ffmpeg
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

def analyze_video(video_path):
    """Analyze video with optimized BLIP processing"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(frame_count // 15, 1)  # Reduced frame sampling
    
    frames = []
    try:
        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Resize frame for faster processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(frame_rgb, (256, 256))  # Smaller size
                frames.append(Image.fromarray(small_frame))
    finally:
        cap.release()
    
    if not frames:
        return ""
    
    # Batch processing with BLIP
    outputs = st.session_state.model(frames)
    captions = [out[0]['generated_text'] for out in outputs]
    
    return " ".join(list(set(captions)))

def generate_audio(description, duration):
    """Optimized audio generation"""
    inputs = st.session_state.processor(
        text=[description],
        padding=True,
        return_tensors="pt",
    ).to(st.session_state.device)  # Move to GPU if available
    
    # Use cached model parameters
    if not hasattr(st.session_state, 'generation_params_set'):
        st.session_state.music_gen.set_generation_params(
            duration=duration,
            use_sampling=True,
            top_k=250
        )
        st.session_state.generation_params_set = True
    
    audio = st.session_state.music_gen.generate(**inputs)
    return audio[0][0].cpu().numpy()

def process_video(uploaded_file):
    """Optimized FFmpeg processing with in-memory streams"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Get duration (max 10s)
        probe = ffmpeg.probe(video_path)
        duration = min(float(probe['format']['duration']), 10)
        
        # Analyze video
        description = analyze_video(video_path)
        
        # Generate audio
        audio_array = generate_audio(description, duration)
        
        # Merge using pipes
        audio_pipe = ffmpeg.input('pipe:', format='f32le', ac=1, ar=32000)
        video_pipe = ffmpeg.input(video_path)
        
        output_path = os.path.join(tmp_dir, "output.mp4")
        (
            ffmpeg
            .concat(video_pipe, audio_pipe, v=1, a=1)
            .output(
                output_path,
                vcodec='copy',
                acodec='aac',
                shortest=None,
                movflags='+faststart'  # For faster streaming
            )
            .run_async(pipe_stdin=True, quiet=True)
        ).stdin.write(audio_array.tobytes())
        
        # Wait for FFmpeg process to complete
        ffmpeg_process = ffmpeg.get_current_process()
        ffmpeg_process.communicate()
        
        # Return as bytes
        with open(output_path, 'rb') as f:
            return f.read()

# Streamlit UI
st.set_page_config(page_title="Video Sound FX", layout="wide")
st.title("🎬 Next-Gen Video Sound Generator")

# Initialize models at startup
if 'initialized' not in st.session_state:
    st.session_state.model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )
    st.session_state.music_gen = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small"
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    st.session_state.processor = AutoProcessor.from_pretrained(
        "facebook/musicgen-small"
    )
    st.session_state.device = next(st.session_state.music_gen.parameters()).device
    st.session_state.initialized = True

uploaded_file = st.file_uploader("Upload video (MP4, max 10s)", type=["mp4"])

if uploaded_file and st.button("Generate Soundtrack"):
    with st.spinner("Processing..."):
        try:
            output_bytes = process_video(uploaded_file)
            
            st.success("Processing Complete!")
            st.video(output_bytes)
            
            st.download_button(
                "Download Enhanced Video",
                data=output_bytes,
                file_name="video_with_sound.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

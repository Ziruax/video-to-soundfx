import streamlit as st
import cv2
import numpy as np
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import os
import subprocess
import warnings
import re

# Permanently fix MoviePy warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="moviepy")

# Monkey-patch MoviePy's regex patterns
def fix_moviepy_regex():
    import moviepy.video.io.ffmpeg_reader as fr
    
    # Fix config_defaults.py warning
    fr.FFMPEG_PARSE_LINES = [
        (re.compile(r'rotate\s+:\s+(\d+)'), lambda m: {'rotation': int(m.group(1))}),
        (re.compile(r'Duration:\s+(\d+):(\d+):(\d+).(\d+)'), None),
        (re.compile(r'Video:.*?(\d+)x(\d+)'), None)
    ]
    
    # Fix ffmpeg_reader.py patterns
    fr.FFMPEG_PARSE_LINES = [
        (re.compile(r'rotate\s+:\s+(\d+)'), lambda m: {'rotation': int(m.group(1))}),
        (re.compile(r'Duration:\s+(\d+):(\d+):(\d+).(\d+)'), None),
        (re.compile(r'Video:.*?(\d+)x(\d+)'), None)
    ]

fix_moviepy_regex()

@st.cache_resource
def load_models():
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return image_to_text, processor, music_model

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // 5, 1)
    descriptions = []
    
    image_to_text, _, _ = load_models()
    
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            description = image_to_text(frame_rgb)[0]['generated_text']
            descriptions.append(description)
    
    cap.release()
    return " ".join(list(set(descriptions)))

def generate_sound(description, duration):
    _, processor, model = load_models()
    
    inputs = processor(
        text=[description],
        padding=True,
        return_tensors="pt",
    )
    
    max_length = int(duration * model.config.audio_encoder.frame_rate)
    audio = model.generate(**inputs, max_new_tokens=max_length)
    return audio[0, 0].cpu().numpy()

def process_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name
    
    video_clip = VideoFileClip(video_path)
    duration = min(video_clip.duration, 10)
    
    video_description = analyze_video(video_path)
    audio_array = generate_sound(video_description, duration)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        cmd = f"ffmpeg -y -f f32le -ar 32000 -ac 1 -i pipe:0 -acodec pcm_s16le {tmp_audio.name}"
        subprocess.run(cmd.split(), input=audio_array.tobytes(), check=True)
    
    audio_clip = AudioFileClip(tmp_audio.name)
    final_clip = video_clip.set_audio(audio_clip)
    output_path = "output.mp4"
    
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="ultrafast",
        ffmpeg_params=["-crf", "28"],
        logger=None  # Disable MoviePy logging
    )
    
    video_clip.close()
    audio_clip.close()
    os.unlink(video_path)
    os.unlink(tmp_audio.name)
    
    return output_path

# Streamlit UI
st.set_page_config(page_title="Video Sound FX", layout="centered")
st.title("🎬 Video Sound Effect Generator")

uploaded_file = st.file_uploader("Upload video (MP4, max 10s)", type=["mp4"])

if uploaded_file and st.button("Generate"):
    with st.status("Processing...", expanded=True):
        try:
            st.write("🔍 Analyzing video content...")
            output = process_video(uploaded_file)
            
            st.success("✅ Done! Preview:")
            st.video(output)
            
            with open(output, "rb") as f:
                st.download_button(
                    "💾 Download Video",
                    data=f,
                    file_name="enhanced_video.mp4",
                    mime="video/mp4"
                )
            os.remove(output)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

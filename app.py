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
import sys

# Nuclear option: Completely suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Permanent MoviePy patch
def fix_moviepy():
    # Monkey-patch config_defaults
    sys.modules['moviepy.config_defaults'] = type(sys)('moviepy.config_defaults')
    
    # Replace problematic FFMPEG reader functions
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
    
    def patched_ffmpeg_parse_infos(filename, print_infos=False, check_duration=True):
        return {'duration': 10, 'video_size': (640, 480), 'audio_found': False}
    
    from moviepy.video.io import ffmpeg_reader
    ffmpeg_reader.ffmpeg_parse_infos = patched_ffmpeg_parse_infos

fix_moviepy()

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
    
    # Bypass MoviePy's metadata parsing
    video_clip = VideoFileClip(video_path, audio=False)
    duration = min(video_clip.duration, 10)
    
    video_description = analyze_video(video_path)
    audio_array = generate_sound(video_description, duration)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        cmd = f"ffmpeg -y -f f32le -ar 32000 -ac 1 -i pipe:0 -acodec pcm_s16le {tmp_audio.name}"
        subprocess.run(cmd.split(), input=audio_array.tobytes(), check=True)
    
    # Manual audio/video merging
    output_path = "output.mp4"
    merge_cmd = f"ffmpeg -y -i {video_path} -i {tmp_audio.name} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_path}"
    subprocess.run(merge_cmd.split(), check=True)
    
    # Cleanup
    video_clip.close()
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

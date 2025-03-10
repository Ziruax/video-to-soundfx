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

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

# Improved MoviePy patch
def patch_moviepy():
    try:
        from moviepy.video.io import ffmpeg_reader
        
        # Define safer regex patterns
        ffmpeg_reader.FFMPEG_VIDEO_INFO_PATTERN = re.compile(
            r'Video:.*?(?P<width>\d+)x(?P<height>\d+)'
        )
        ffmpeg_reader.FFMPEG_ROTATION_PATTERN = re.compile(
            r'rotate\s+:\s+(?P<rotation>\d+)'
        )
        ffmpeg_reader.FFMPEG_DURATION_PATTERN = re.compile(
            r'Duration:\s+(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)\.(?P<ms>\d+)'
        )
    except ImportError:
        pass  # Fallback if patching fails

patch_moviepy()

@st.cache_resource
def load_models():
    try:
        image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        return image_to_text, processor, music_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None

def analyze_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(total_frames // 5, 1)
        descriptions = []
        
        image_to_text, _, _ = load_models()
        if image_to_text is None:
            raise ValueError("Image-to-text model not loaded")
            
        for i in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                description = image_to_text(frame_rgb)[0]['generated_text']
                descriptions.append(description)
        
        cap.release()
        return " ".join(list(set(descriptions)))
    except Exception as e:
        st.error(f"Video analysis failed: {str(e)}")
        return ""

def generate_sound(description, duration):
    try:
        _, processor, model = load_models()
        if processor is None or model is None:
            raise ValueError("Audio generation models not loaded")
            
        inputs = processor(
            text=[description],
            padding=True,
            return_tensors="pt",
        )
        
        max_length = int(duration * model.config.audio_encoder.frame_rate)
        audio = model.generate(**inputs, max_new_tokens=max_length)
        return audio[0, 0].cpu().numpy()
    except Exception as e:
        st.error(f"Sound generation failed: {str(e)}")
        return None

def process_video(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(uploaded_file.read())
            video_path = tmp_video.name
        
        video_clip = VideoFileClip(video_path)
        duration = min(video_clip.duration, 10)
        
        st.write("Analyzing video frames...")
        video_description = analyze_video(video_path)
        if not video_description:
            raise ValueError("No video description generated")
            
        st.write("Generating sound effects...")
        audio_array = generate_sound(video_description, duration)
        if audio_array is None:
            raise ValueError("Audio generation failed")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            cmd = [
                "ffmpeg", "-y",
                "-f", "f32le",
                "-ar", "32000",
                "-ac", "1",
                "-i", "pipe:0",
                "-acodec", "pcm_s16le",
                tmp_audio.name
            ]
            subprocess.run(cmd, input=audio_array.tobytes(), check=True, capture_output=True)
        
        audio_clip = AudioFileClip(tmp_audio.name)
        final_clip = video_clip.set_audio(audio_clip)
        
        output_path = tempfile.mktemp(suffix=".mp4")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=2,
            preset="ultrafast",
            ffmpeg_params=["-crf", "28"],
            logger=None,
            temp_audiofile=tempfile.mktemp(suffix=".wav")
        )
        
        video_clip.close()
        audio_clip.close()
        os.unlink(video_path)
        os.unlink(tmp_audio.name)
        
        return output_path
    except Exception as e:
        st.error(f"Video processing failed: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Video Sound FX", layout="centered")
st.title("🎬 Video Sound Effect Generator")

uploaded_file = st.file_uploader("Upload video (MP4, max 10s)", type=["mp4"])

if uploaded_file and st.button("Generate"):
    with st.status("Processing...", expanded=True):
        output = process_video(uploaded_file)
        if output:
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

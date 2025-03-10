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

# Suppress only SyntaxWarnings from MoviePy
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Patch MoviePy to fix regex-related warnings
def patch_moviepy():
    import moviepy.video.io.ffmpeg_reader as ffmpeg_reader
    # Redefine problematic regex patterns with proper escaping
    ffmpeg_reader.FFMPEG_VIDEO_INFO_PATTERN = re.compile(r"Video:.*?(?P<width>\\d+)x(?P<height>\\d+)")
    ffmpeg_reader.FFMPEG_ROTATION_PATTERN = re.compile(r"rotate\s+:\s+(?P<rotation>\\d+)")
    ffmpeg_reader.FFMPEG_DURATION_PATTERN = re.compile(
        r"Duration:\s+(?P<hours>\\d+):(?P<minutes>\\d+):(?P<seconds>\\d+)\.(?P<ms>\\d+)"
    )

# Apply the patch at startup
patch_moviepy()

# Cache models to load only once
@st.cache_resource
def load_models():
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return image_to_text, processor, music_model

# Analyze video frames and generate a description
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // 5, 1)  # Analyze 5 frames
    descriptions = []
    
    image_to_text, _, _ = load_models()
    
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            description = image_to_text(frame_rgb)[0]["generated_text"]
            descriptions.append(description)
    
    cap.release()
    return " ".join(list(set(descriptions)))  # Unique descriptions

# Generate audio based on video description
def generate_sound(description, duration):
    _, processor, model = load_models()
    inputs = processor(text=[description], padding=True, return_tensors="pt")
    max_length = int(duration * model.config.audio_encoder.frame_rate)
    audio = model.generate(**inputs, max_new_tokens=max_length)
    return audio[0, 0].cpu().numpy()

# Process the uploaded video
def process_video(uploaded_file):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name
    
    # Load and limit video duration
    video_clip = VideoFileClip(video_path)
    duration = min(video_clip.duration, 10)  # Cap at 10 seconds
    
    # Analyze video and generate sound
    st.write("🔍 Analyzing video frames...")
    video_description = analyze_video(video_path)
    st.write("🎶 Generating sound effects...")
    audio_array = generate_sound(video_description, duration)
    
    # Write audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        cmd = [
            "ffmpeg", "-y", "-f", "f32le", "-ar", "32000", "-ac", "1",
            "-i", "pipe:0", "-acodec", "pcm_s16le", tmp_audio.name
        ]
        subprocess.run(cmd, input=audio_array.tobytes(), check=True)
    
    # Combine video and audio
    audio_clip = AudioFileClip(tmp_audio.name)
    final_clip = video_clip.set_audio(audio_clip)
    output_path = "output.mp4"
    
    st.write("🎥 Finalizing video...")
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="ultrafast",
        ffmpeg_params=["-crf", "28"],  # Higher CRF for smaller file size
        logger=None
    )
    
    # Clean up temporary files
    video_clip.close()
    audio_clip.close()
    os.unlink(video_path)
    os.unlink(tmp_audio.name)
    
    return output_path

# Streamlit UI
st.set_page_config(page_title="Video Sound FX", layout="centered")
st.title("🎬 Video Sound Effect Generator")
st.write("Upload a short video (MP4, max 10s) to add AI-generated sound effects!")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

if uploaded_file and st.button("Generate"):
    with st.spinner("Processing your video..."):
        try:
            output_path = process_video(uploaded_file)
            st.success("✅ Video processed successfully!")
            
            # Display the result
            st.video(output_path)
            
            # Offer download option
            with open(output_path, "rb") as f:
                st.download_button(
                    label="💾 Download Video",
                    data=f,
                    file_name="enhanced_video.mp4",
                    mime="video/mp4"
                )
            os.remove(output_path)  # Clean up
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.write("Please try again or check the error details.")

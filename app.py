import streamlit as st
import cv2
import numpy as np
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import os
import subprocess

# Cache models
@st.cache_resource
def load_models():
    # Image analysis model
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    # Music generation model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    
    return image_to_text, processor, music_model

def analyze_video_frames(video_path):
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

def generate_music_from_text(description, duration):
    _, processor, model = load_models()
    
    inputs = processor(
        text=[description],
        padding=True,
        return_tensors="pt",
    )
    
    # Convert duration to seconds (max 10s)
    max_length = int(duration * model.config.audio_encoder.frame_rate)
    
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_length)
    return audio_values[0, 0].cpu().numpy()

def process_video(uploaded_file):
    # Save temp video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_file.read())
        video_path = tmp_video.name
    
    # Get video duration
    video_clip = VideoFileClip(video_path)
    duration = min(video_clip.duration, 10)
    
    # Analyze video content
    video_description = analyze_video_frames(video_path)
    
    # Generate audio
    audio_array = generate_music_from_text(video_description, duration)
    
    # Save and combine audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        # Convert to proper audio format
        cmd = f"ffmpeg -y -f f32le -ar 32000 -ac 1 -i pipe:0 -acodec pcm_s16le {tmp_audio.name}"
        subprocess.run(cmd.split(), input=audio_array.tobytes(), check=True)
    
    # Merge audio with video
    audio_clip = AudioFileClip(tmp_audio.name)
    final_clip = video_clip.set_audio(audio_clip)
    
    # Export final video
    output_path = "output.mp4"
    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        preset="ultrafast",
        ffmpeg_params=["-crf", "28"]
    )
    
    # Cleanup
    video_clip.close()
    audio_clip.close()
    os.unlink(video_path)
    os.unlink(tmp_audio.name)
    
    return output_path

# Streamlit UI
st.set_page_config(page_title="Video Sound FX Generator", layout="wide")
st.title("🎥 AI Video Sound Effects Generator")

uploaded_file = st.file_uploader("Upload a short video (max 10 seconds)", type=["mp4", "mov"])

if uploaded_file:
    if st.button("Generate Enhanced Video"):
        with st.status("Processing...", expanded=True):
            try:
                st.write("⏳ Analyzing video content...")
                output_path = process_video(uploaded_file)
                
                st.success("✅ Processing complete!")
                st.video(output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        "💾 Download Enhanced Video",
                        data=f,
                        file_name="video_with_soundfx.mp4",
                        mime="video/mp4"
                    )
                os.remove(output_path)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
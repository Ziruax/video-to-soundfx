import streamlit as st
import imageio
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, MusicgenForConditionalGeneration
import soundfile as sf
import torch
import os
import tempfile

# Try importing moviepy with fallback
try:
    import moviepy.editor as mpy
except ModuleNotFoundError:
    st.error("The 'moviepy' library is not installed. Please ensure 'moviepy==1.0.3' is in requirements.txt and installed.")
    st.stop()

# Set page title and instructions
st.title("Story Video Sound Effect Sync Generator")
st.write("Upload an MP4 video to auto-generate and sync a high-quality sound effect.")

# User-configurable settings
num_frames_to_extract = st.slider("Number of frames to analyze", 1, 3, 1, help="Fewer frames = faster processing")
mix_original_audio = st.checkbox("Mix with original audio", value=False, help="Blend sound effect with video’s original sound")

# Enhanced prompt generation function
def enhance_prompt(base_description):
    """Generate a detailed, sound-specific prompt from BLIP caption."""
    base = base_description.lower().strip()
    
    # Define action, object, and environment keywords
    actions = {
        "walk": "crisp footsteps on a wooden floor",
        "run": "rapid footsteps and heavy breathing",
        "drive": "engine roar and tires screeching",
        "talk": "soft voices and background murmur",
        "crash": "loud crash and debris scattering",
        "fall": "thud of impact and rustling debris"
    }
    objects = {
        "person": "human activity with subtle breathing",
        "dog": "playful barks and pawsteps",
        "car": "mechanical hum and tire friction",
        "tree": "rustling leaves in a breeze",
        "forest": "gentle wind and distant bird calls"
    }
    environments = {
        "room": "echoing footsteps and muffled sounds",
        "street": "distant traffic and urban hum",
        "forest": "wind through trees and twigs snapping",
        "outside": "open air with faint wind"
    }

    # Extract key elements from the caption
    sound_description = ""
    for action, sound in actions.items():
        if action in base:
            sound_description = sound
            break
    if not sound_description:  # Default to subtle ambient if no action
        sound_description = "subtle ambient hum"

    # Add object-specific sounds
    for obj, sound in objects.items():
        if obj in base:
            sound_description += f" and {sound}"
            break

    # Add environment if detected
    for env, sound in environments.items():
        if env in base:
            sound_description += f" in a {env} with {sound}"
            break

    # Construct final prompt
    return f"{base} with {sound_description}"

# File uploader for video
uploaded_file = st.file_uploader("Upload an MP4 video (high resolution)", type=["mp4"])

if uploaded_file is not None:
    try:
        # Temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.getbuffer())
            temp_video_path = temp_video.name

        # Progress bar setup
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Extract frames
        status_text.text("Extracting frames...")
        video = imageio.get_reader(temp_video_path, "ffmpeg")
        total_frames = len(list(video.iter_data()))
        step = max(1, total_frames // num_frames_to_extract)
        frames = [
            Image.fromarray(video.get_data(i)) 
            for i in range(0, min(total_frames, num_frames_to_extract * step), step)
        ][:num_frames_to_extract]
        progress_bar.progress(20)

        # Load BLIP model
        @st.cache_resource
        def load_blip_model():
            processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            if torch.cuda.is_available():
                model = model.half().to("cuda")
            return processor, model

        processor, model = load_blip_model()

        # Generate and enhance text descriptions
        status_text.text("Analyzing frames...")
        descriptions = []
        for i, frame in enumerate(frames):
            inputs = processor(images=frame, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = model.generate(**inputs)
            base_description = processor.decode(out[0], skip_special_tokens=True)
            enhanced_description = enhance_prompt(base_description)
            descriptions.append(enhanced_description)
            progress_bar.progress(20 + int(30 * (i + 1) / len(frames)))

        text_prompt = ". ".join(descriptions)
        st.write("Enhanced text prompt:", text_prompt)

        # Load MusicGen model
        @st.cache_resource
        def load_musicgen_model():
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            if torch.cuda.is_available():
                model = model.half().to("cuda")
            return processor, model

        musicgen_processor, musicgen_model = load_musicgen_model()

        # Generate sound effect (~8 seconds)
        status_text.text("Generating sound effect...")
        inputs = musicgen_processor(
            text=[text_prompt],
            padding=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        audio_values = musicgen_model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=True, 
            guidance_scale=3.0, 
            top_k=50, 
            top_p=0.95
        )
        audio_array = audio_values[0].cpu().numpy()
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        audio_array = audio_array / np.max(np.abs(audio_array)) * 0.9
        audio_array = np.clip(audio_array, -1.0, 1.0)
        sample_rate = 32000
        progress_bar.progress(60)

        # Save temporary audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, audio_array, sample_rate)
            temp_audio_path = temp_audio.name

        # Synchronize with video using mpy
        status_text.text("Syncing audio with video...")
        video_clip = mpy.VideoFileClip(temp_video_path)
        video_duration = video_clip.duration
        audio_clip = mpy.AudioFileClip(temp_audio_path)

        # Adjust audio length
        if audio_clip.duration < video_duration:
            loops_needed = int(np.ceil(video_duration / audio_clip.duration))
            audio_clip = mpy.concatenate_audioclips([audio_clip] * loops_needed).subclip(0, video_duration)
        else:
            audio_clip = audio_clip.subclip(0, video_duration)

        # Mix or replace audio
        if mix_original_audio and video_clip.audio:
            final_audio = video_clip.audio.volumex(0.5) + audio_clip.volumex(0.5)
        else:
            final_audio = audio_clip

        # Set audio to video
        final_video = video_clip.set_audio(final_audio)

        # Save final video with high quality
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            preset="medium",  # Better quality than ultrafast
            bitrate="8000k",  # Higher bitrate for video quality
            audio_bitrate="192k",  # Good audio quality
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )
        progress_bar.progress(90)

        # Provide playback and download
        status_text.text("Done!")
        st.video(output_path)
        with open(output_path, "rb") as video_file:
            st.download_button(
                label="Download Synced Video",
                data=video_file,
                file_name="synced_story_video.mp4",
                mime="video/mp4"
            )
        progress_bar.progress(100)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Try reducing frames or uploading a smaller video.")

    finally:
        # Clean up
        for path in [temp_video_path, temp_audio_path, output_path]:
            if 'path' in locals() and os.path.exists(path):
                os.remove(path)

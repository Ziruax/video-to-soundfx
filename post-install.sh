#!/bin/bash
# Completely remove moviepy
pip uninstall -y moviepy

# Alternative video processing
pip install imageio imageio-ffmpeg

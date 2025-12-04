#!/usr/bin/env python3
"""
Generate the podcast disclaimer audio.

This is a one-time generation script. Run it once to create the disclaimer,
then it gets included in all future episodes automatically.

Uses Chatterbox on Replicate with the 'crotchety-guy' voice sample for
a dry, serious British-style delivery.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

import replicate
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
VOICES_DIR = PROJECT_ROOT / "config" / "voices"
OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "show-elements" / "mixed"

# Voice sample for disclaimer (crotchety-guy for dry, serious delivery)
DISCLAIMER_VOICE = VOICES_DIR / "crotchety-guy" / "wav" / "crotchety-guy-1min.wav"

# The disclaimer text (dry British humor style)
DISCLAIMER_TEXT = """My Weird Prompts is an AI-generated podcast produced by Daniel Rosehill,
a verified human with documentation to prove it. While every effort is made to ensure
our AI hosts draw from credible sources and run on cutting-edge models, they do
occasionally make things up with alarming confidence. As with everything in life,
and especially this podcast, treat all claims as suspicious until independently verified.
The management accepts no responsibility for any decisions made based on the confident
ramblings of our artificial presenters. Now then, on with the show."""


def generate_disclaimer():
    """Generate the disclaimer audio using Chatterbox."""
    print("Generating podcast disclaimer audio...")
    print(f"Voice sample: {DISCLAIMER_VOICE}")
    print(f"Output: {OUTPUT_DIR / 'disclaimer.mp3'}")

    # Verify API token
    api_token = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable not set")
    os.environ["REPLICATE_API_TOKEN"] = api_token

    # Verify voice sample exists
    if not DISCLAIMER_VOICE.exists():
        raise FileNotFoundError(f"Voice sample not found: {DISCLAIMER_VOICE}")

    # Upload voice sample
    print("Uploading voice sample...")
    voice_file = replicate.files.create(
        file=DISCLAIMER_VOICE,
        metadata={"purpose": "disclaimer_voice"}
    )
    voice_url = voice_file.urls['get']
    print(f"  Uploaded: {voice_file.id}")

    # Generate audio
    print("Generating TTS audio...")
    output = replicate.run(
        "resemble-ai/chatterbox:1b8422bc49635c20d0a84e387ed20879c0dd09254ecdb4e75dc4bec10ff94e97",
        input={
            "prompt": DISCLAIMER_TEXT,
            "audio_prompt": voice_url,
            "exaggeration": 0.3,  # Less exaggeration for serious delivery
            "cfg_weight": 0.5,
        }
    )

    # Download output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_DIR / "disclaimer_raw.wav"
    output_path = OUTPUT_DIR / "disclaimer.mp3"

    # Handle various output types from Replicate
    if hasattr(output, 'read'):
        # File-like object
        with open(temp_path, 'wb') as f:
            f.write(output.read())
    else:
        # URL string or FileOutput object
        output_url = output.url if hasattr(output, 'url') else str(output)
        print(f"  Downloading from: {output_url[:80]}...")
        urllib.request.urlretrieve(output_url, str(temp_path))

    print(f"  Raw audio saved: {temp_path}")
    print(f"  File size: {temp_path.stat().st_size} bytes")

    # Check file type
    file_type_result = subprocess.run(["file", str(temp_path)], capture_output=True, text=True)
    print(f"  File type: {file_type_result.stdout.strip()}")

    # Normalize and convert to MP3
    print("Normalizing and encoding to MP3...")
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

    # Cleanup temp file
    temp_path.unlink()

    # Get duration
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)
    ]
    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
    duration = float(duration_result.stdout.strip())

    print(f"\nDisclaimer generated successfully!")
    print(f"  File: {output_path}")
    print(f"  Duration: {duration:.1f}s")
    print(f"\nThis will be automatically included after the intro in all future episodes.")


if __name__ == "__main__":
    generate_disclaimer()

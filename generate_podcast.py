#!/usr/bin/env python3
"""
AI Podcast Generator

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to Gemini to transcribe and generate a podcast dialogue script
3. Converts script to multi-speaker audio via Gemini TTS
4. Concatenates: intro jingle + user audio + AI response + outro jingle

Requires:
    pip install google-genai python-dotenv

Environment:
    GEMINI_API_KEY - Your Gemini API key (can be in .env file)
"""

import os
import struct
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent
AUDIO_FILES_DIR = PROJECT_ROOT / "audio-files"
PROMPTS_DIR = AUDIO_FILES_DIR / "prompt"
RESPONSES_DIR = AUDIO_FILES_DIR / "responses"
EPISODES_DIR = AUDIO_FILES_DIR / "episodes"
JINGLES_DIR = AUDIO_FILES_DIR / "jingles"

# Ensure output directories exist
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

# Podcast configuration
PODCAST_NAME = "AI Conversations"
HOST_NAME = "Daniel"
AI_NAME = "Claude"

# Voice configuration for TTS
HOST_VOICE = "Orus"      # Firm voice for host
AI_VOICE = "Puck"        # Upbeat voice for AI

# System prompt for generating the podcast script
PODCAST_SCRIPT_PROMPT = """You are a podcast script writer. You will receive an audio recording from the host of a podcast.

Your task:
1. Listen to and understand the host's prompt/question
2. Generate a natural podcast dialogue response

Output format - Generate a script in this exact format:

{ai_name}: [Your thoughtful, conversational response to the host's question. Be informative but conversational, as if speaking on a podcast. Keep responses focused and engaging - aim for 1-3 paragraphs of spoken content.]

Guidelines:
- Be conversational and natural, like a real podcast co-host
- Provide substantive, helpful responses
- Don't be overly formal or robotic
- Match the energy and tone of the host
- If the host asks a technical question, explain clearly but accessibly
""".format(ai_name=AI_NAME)


def get_client() -> genai.Client:
    """Initialize Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Add it to .env file.")
    return genai.Client(api_key=api_key)


def transcribe_and_generate_script(client: genai.Client, audio_path: Path) -> str:
    """
    Send audio to Gemini, transcribe it, and generate a podcast script response.

    Args:
        client: Gemini client
        audio_path: Path to the audio file

    Returns:
        Generated podcast script text
    """
    print(f"Uploading audio file: {audio_path}")

    # Upload the audio file
    audio_file = client.files.upload(file=str(audio_path))
    print(f"Uploaded file: {audio_file.name}")

    # Generate the script
    print("Generating podcast script...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PODCAST_SCRIPT_PROMPT, audio_file]
    )

    script = response.text
    print(f"Generated script:\n{'-'*40}\n{script}\n{'-'*40}")

    return script


def parse_audio_mime_type(mime_type: str) -> dict:
    """Parse bits per sample and rate from audio MIME type."""
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Convert raw audio data to WAV format with proper header."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data


def generate_tts_audio(client: genai.Client, script: str, output_path: Path) -> Path:
    """
    Convert podcast script to audio using Gemini TTS.

    Args:
        client: Gemini client
        script: The podcast script text
        output_path: Where to save the audio

    Returns:
        Path to the generated audio file
    """
    print("Generating TTS audio...")

    # For single-speaker (just AI response), use single speaker config
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=AI_VOICE,
                    )
                )
            ),
        )
    )

    # Extract audio data
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    mime_type = response.candidates[0].content.parts[0].inline_data.mime_type

    # Convert to WAV
    wav_data = convert_to_wav(audio_data, mime_type)

    # Save the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(wav_data)

    print(f"TTS audio saved to: {output_path}")
    return output_path


def concatenate_audio(
    output_path: Path,
    user_audio: Path,
    ai_audio: Path,
    intro_jingle: Path = None,
    outro_jingle: Path = None,
) -> Path:
    """
    Concatenate audio segments into final podcast episode using ffmpeg.

    Order: intro (optional) -> user audio -> AI response -> outro (optional)

    Args:
        output_path: Where to save the final episode
        user_audio: Path to user's audio prompt
        ai_audio: Path to AI-generated response audio
        intro_jingle: Optional intro jingle
        outro_jingle: Optional outro jingle

    Returns:
        Path to the final episode
    """
    print("Concatenating audio segments...")

    # Build list of audio files to concatenate
    audio_files = []
    if intro_jingle and intro_jingle.exists():
        audio_files.append(intro_jingle)
    audio_files.append(user_audio)
    audio_files.append(ai_audio)
    if outro_jingle and outro_jingle.exists():
        audio_files.append(outro_jingle)

    # Create a temporary file list for ffmpeg concat
    filelist_path = output_path.parent / "filelist.txt"

    # We need to normalize all audio to the same format first
    normalized_files = []
    for i, audio_file in enumerate(audio_files):
        normalized_path = output_path.parent / f"temp_normalized_{i}.wav"
        # Normalize to 44.1kHz, mono, 16-bit
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_file),
            "-ar", "44100", "-ac", "1", "-sample_fmt", "s16",
            str(normalized_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        normalized_files.append(normalized_path)

    # Create file list for concat
    with open(filelist_path, "w") as f:
        for nf in normalized_files:
            f.write(f"file '{nf}'\n")

    # Concatenate using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c:a", "libmp3lame", "-b:a", "128k",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Cleanup temp files
    filelist_path.unlink()
    for nf in normalized_files:
        nf.unlink()

    print(f"Final episode saved to: {output_path}")
    return output_path


def generate_podcast_episode(
    prompt_audio_path: Path,
    episode_name: str = None,
) -> Path:
    """
    Generate a complete podcast episode from a user's audio prompt.

    Args:
        prompt_audio_path: Path to the user's recorded prompt
        episode_name: Optional name for the episode

    Returns:
        Path to the final episode MP3
    """
    if episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{timestamp}"

    print(f"\n{'='*50}")
    print(f"Generating podcast episode: {episode_name}")
    print(f"{'='*50}\n")

    # Initialize client
    client = get_client()

    # Step 1: Transcribe and generate script
    script = transcribe_and_generate_script(client, prompt_audio_path)

    # Save the script for reference
    script_path = RESPONSES_DIR / f"{episode_name}_script.txt"
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Script saved to: {script_path}")

    # Step 2: Generate TTS audio
    ai_audio_path = RESPONSES_DIR / f"{episode_name}_response.wav"
    generate_tts_audio(client, script, ai_audio_path)

    # Step 3: Concatenate into final episode
    episode_path = EPISODES_DIR / f"{episode_name}.mp3"
    intro_jingle = JINGLES_DIR / "intro.mp3"
    outro_jingle = JINGLES_DIR / "outro.mp3"

    concatenate_audio(
        output_path=episode_path,
        user_audio=prompt_audio_path,
        ai_audio=ai_audio_path,
        intro_jingle=intro_jingle if intro_jingle.exists() else None,
        outro_jingle=outro_jingle if outro_jingle.exists() else None,
    )

    print(f"\n{'='*50}")
    print(f"Episode generated successfully!")
    print(f"Output: {episode_path}")
    print(f"{'='*50}\n")

    return episode_path


def main():
    """Main entry point."""
    # Default to the test prompt if no argument provided
    if len(sys.argv) > 1:
        prompt_path = Path(sys.argv[1])
    else:
        prompt_path = PROMPTS_DIR / "raw.mp3"

    if not prompt_path.exists():
        print(f"Error: Audio file not found: {prompt_path}")
        sys.exit(1)

    episode_path = generate_podcast_episode(prompt_path)
    print(f"Done! Episode saved to: {episode_path}")


if __name__ == "__main__":
    main()

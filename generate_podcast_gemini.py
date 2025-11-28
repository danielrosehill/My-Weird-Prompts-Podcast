#!/usr/bin/env python3
"""
AI Podcast Generator (Gemini Native TTS Version)

Uses Gemini's native multispeaker TTS capability (gemini-2.5-pro-preview-tts)
for higher quality, more natural dialogue generation.

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to Gemini to transcribe and generate a diarized podcast dialogue script (~15 min)
3. Converts script to multi-speaker audio via Gemini's native TTS
4. Concatenates: intro jingle + AI dialogue + outro jingle

Requires:
    pip install google-genai python-dotenv

Environment:
    GEMINI_API_KEY - Your Gemini API key (can be in .env file)
"""

import json
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent

# Queue-based directory structure
PROMPTS_TO_PROCESS_DIR = PROJECT_ROOT / "prompts" / "to-process"
PROMPTS_DONE_DIR = PROJECT_ROOT / "prompts" / "done"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
EPISODES_DIR = OUTPUT_DIR / "episodes"
JINGLES_DIR = PROJECT_ROOT / "show-elements" / "mixed"

# Ensure output directories exist
EPISODES_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DONE_DIR.mkdir(parents=True, exist_ok=True)

# Podcast configuration
PODCAST_NAME = "AI Conversations"
HOST_NAME = "Herman"     # Will use Charon voice - warm, friendly
CO_HOST_NAME = "Donald"  # Will use Orus voice - knowledgeable, authoritative

# Gemini TTS voice mapping (both male voices)
SPEAKER_VOICE_MAP = {
    HOST_NAME: "Charon",    # Male, warm, welcoming tone
    CO_HOST_NAME: "Orus",   # Male, clear, authoritative tone
}

# Target episode length (~15 minutes at ~150 words per minute)
TARGET_WORD_COUNT = 2250  # ~15 minutes of dialogue

# System prompt for generating a diarized podcast dialogue script (~15 minutes)
PODCAST_SCRIPT_PROMPT = """You are a podcast script writer creating an engaging two-host dialogue for "{podcast_name}".

The user has recorded an audio prompt with a topic/question. Listen carefully and generate a comprehensive ~15 minute podcast episode script as a natural conversation between two AI hosts.

## The Hosts

- **{host_name}**: The curious, enthusiastic host who asks probing questions, shares relatable examples, and keeps the conversation accessible. Tends to think out loud and make connections to everyday life.

- **{co_host_name}**: The knowledgeable expert who provides deep insights, technical details, and authoritative explanations. Balances depth with clarity, using analogies to explain complex topics.

## Script Format

You MUST output the script in this exact diarized format - each line starting with the speaker name followed by a colon:

{host_name}: [dialogue]
{co_host_name}: [dialogue]
{host_name}: [dialogue]
...

## Episode Structure (~15 minutes total when spoken, approximately 2000-2500 words)

1. **Opening Hook** (30 seconds)
   - {host_name} introduces the topic with an intriguing angle
   - {co_host_name} adds a surprising fact or stakes

2. **Topic Introduction** (2 minutes)
   - Both hosts establish what they'll cover
   - Set up why listeners should care

3. **Core Discussion** (8-10 minutes)
   - Deep, substantive back-and-forth exploration of the topic
   - {co_host_name} provides expert insights with specific details
   - {host_name} asks clarifying questions, plays devil's advocate
   - Include specific examples, data, case studies, historical context
   - Natural tangents that add value
   - Multiple sub-topics within the main theme

4. **Practical Takeaways** (2-3 minutes)
   - What can listeners actually do with this information?
   - Real-world applications
   - Different perspectives on implementation

5. **Closing Thoughts** (1-2 minutes)
   - Future implications and predictions
   - What questions remain unanswered
   - Tease potential follow-up topics
   - Sign off

## Dialogue Guidelines

- **Natural speech patterns**: Include occasional filler words ("you know", "I mean", "right"), brief pauses indicated by "..." or "hmm", and natural flow
- **Reactions**: "That's fascinating!", "Wait, really?", "Hmm, that's a good point", "Okay so let me make sure I understand..."
- **Length variety**: Mix short reactive lines (1-2 sentences) with longer explanatory passages (3-5 sentences)
- **Chemistry**: The hosts should build on each other's points, occasionally express genuine surprise, and respectfully challenge assumptions
- **Engagement hooks**: "Here's the thing...", "What most people don't realize...", "This is where it gets interesting...", "But here's what blew my mind..."

## Content Requirements

- **Depth**: Provide substantive, educational content - go beyond surface-level. This should feel like a real podcast people learn from.
- **Specificity**: Use real numbers, names, dates, examples when possible
- **Accuracy**: Be precise on technical topics. Mark speculation clearly with phrases like "from what we know" or "current research suggests"
- **Accessibility**: Explain jargon when used, use analogies for complex concepts
- **Length**: AIM FOR 2000-2500 WORDS TOTAL. This is critical for reaching ~15 minutes.

## Output

Generate ONLY the diarized script. No stage directions, no [brackets], no metadata - just speaker names and their dialogue.

Example format:
{host_name}: Welcome back to {podcast_name}! Today we're diving into something that's been all over the headlines lately, and honestly, I've been really curious to dig into this one.
{co_host_name}: Yeah, and I think what's interesting is that most of the coverage has been missing the real story here. There's this whole dimension that people aren't talking about.
{host_name}: Okay, so break it down for us. What's actually going on beneath the surface?

Now generate the full ~15 minute episode script (2000-2500 words) based on the user's audio prompt.
""".format(podcast_name=PODCAST_NAME, host_name=HOST_NAME, co_host_name=CO_HOST_NAME)


def get_gemini_client() -> genai.Client:
    """Initialize Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Add it to .env file.")
    return genai.Client(api_key=api_key)


def transcribe_and_generate_script(client: genai.Client, audio_path: Path) -> str:
    """
    Send audio to Gemini, transcribe it, and generate a diarized podcast dialogue script.

    Args:
        client: Gemini client
        audio_path: Path to the audio file

    Returns:
        Generated diarized podcast script text
    """
    print(f"Uploading audio file: {audio_path}")

    # Upload the audio file
    audio_file = client.files.upload(file=str(audio_path))
    print(f"Uploaded file: {audio_file.name}")

    # Generate the diarized script using a larger model for longer output
    print("Generating diarized podcast script (~15 min episode)...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PODCAST_SCRIPT_PROMPT, audio_file],
        config=types.GenerateContentConfig(
            max_output_tokens=16000,  # Allow for longer scripts
        )
    )

    script = response.text
    word_count = len(script.split())
    print(f"Generated script with ~{word_count} words")
    print(f"Script preview:\n{'-'*40}\n{script[:1500]}...\n{'-'*40}")

    return script


def parse_diarized_script(script: str) -> list[dict]:
    """
    Parse a diarized script into a list of speaker segments.

    Args:
        script: The diarized script text (format: "Speaker: dialogue")

    Returns:
        List of dicts with 'speaker' and 'text' keys
    """
    segments = []
    # Match lines that start with speaker name followed by colon
    pattern = rf'^({HOST_NAME}|{CO_HOST_NAME}):\s*(.+?)(?=^(?:{HOST_NAME}|{CO_HOST_NAME}):|\Z)'

    # Use multiline and dotall flags
    matches = re.findall(pattern, script, re.MULTILINE | re.DOTALL)

    for speaker, text in matches:
        # Clean up the text
        text = text.strip()
        if text:
            segments.append({
                'speaker': speaker,
                'text': text,
                'voice': SPEAKER_VOICE_MAP[speaker]
            })

    # Fallback: line-by-line parsing if regex didn't work well
    if not segments:
        print("Using fallback line-by-line parsing...")
        for line in script.split('\n'):
            line = line.strip()
            if line.startswith(f"{HOST_NAME}:"):
                text = line[len(f"{HOST_NAME}:"):].strip()
                if text:
                    segments.append({
                        'speaker': HOST_NAME,
                        'text': text,
                        'voice': SPEAKER_VOICE_MAP[HOST_NAME]
                    })
            elif line.startswith(f"{CO_HOST_NAME}:"):
                text = line[len(f"{CO_HOST_NAME}:"):].strip()
                if text:
                    segments.append({
                        'speaker': CO_HOST_NAME,
                        'text': text,
                        'voice': SPEAKER_VOICE_MAP[CO_HOST_NAME]
                    })

    print(f"Parsed {len(segments)} dialogue segments")
    return segments


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Generate a WAV file header for the given audio data.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the complete WAV file.
    """
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


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """
    Parse bits per sample and rate from an audio MIME type string.

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys.
    """
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def generate_multispeaker_audio(client: genai.Client, segments: list[dict], output_path: Path) -> Path:
    """
    Generate multi-speaker audio using Gemini's native TTS.

    Args:
        client: Gemini client
        segments: List of parsed dialogue segments
        output_path: Where to save the output WAV file

    Returns:
        Path to the generated audio file
    """
    # Build the diarized text with speaker labels for Gemini TTS
    # Format: "Speaker 1: text\nSpeaker 2: text\n..."
    # We map our speaker names to "Speaker 1" and "Speaker 2" format
    speaker_mapping = {HOST_NAME: "Speaker 1", CO_HOST_NAME: "Speaker 2"}

    diarized_lines = []
    for segment in segments:
        speaker_label = speaker_mapping[segment['speaker']]
        diarized_lines.append(f"{speaker_label}: {segment['text']}")

    full_script = "\n".join(diarized_lines)

    print(f"Generating multi-speaker audio for {len(segments)} segments...")
    print(f"Script length: {len(full_script)} characters")

    # Configure multi-speaker TTS
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Read this podcast conversation aloud naturally, with appropriate pacing and emotion:\n\n{full_script}"
                ),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=SPEAKER_VOICE_MAP[HOST_NAME]
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=SPEAKER_VOICE_MAP[CO_HOST_NAME]
                            )
                        ),
                    ),
                ]
            ),
        ),
    )

    # Collect all audio chunks
    audio_chunks = []
    chunk_count = 0

    print("Streaming TTS generation...")
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-pro-preview-tts",
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            inline_data = part.inline_data
            data_buffer = inline_data.data

            # Convert to WAV if needed
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)

            audio_chunks.append(data_buffer)
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"  Received {chunk_count} audio chunks...")

    print(f"Received {chunk_count} total audio chunks")

    if not audio_chunks:
        raise ValueError("No audio data received from Gemini TTS")

    # Combine chunks and save
    # For WAV files, we need to handle headers properly
    # The first chunk should have the header, subsequent chunks are raw data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save all chunks to temporary files, then concatenate with ffmpeg
    temp_dir = output_path.parent / "_temp_tts"
    temp_dir.mkdir(exist_ok=True)

    temp_files = []
    for i, chunk_data in enumerate(audio_chunks):
        temp_path = temp_dir / f"chunk_{i:04d}.wav"
        with open(temp_path, 'wb') as f:
            f.write(chunk_data)
        temp_files.append(temp_path)

    # Concatenate all chunks using ffmpeg
    filelist_path = temp_dir / "filelist.txt"
    with open(filelist_path, "w") as f:
        for tf in temp_files:
            f.write(f"file '{tf}'\n")

    # Concatenate to final WAV
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Cleanup temp files
    shutil.rmtree(temp_dir)

    print(f"Audio saved to: {output_path}")
    return output_path


def concatenate_episode(
    dialogue_audio: Path,
    output_path: Path,
    user_prompt_audio: Path = None,
    intro_jingle: Path = None,
    outro_jingle: Path = None,
) -> Path:
    """
    Concatenate all episode audio: intro + user prompt + dialogue + outro.

    Args:
        dialogue_audio: Path to the main dialogue audio
        output_path: Where to save the final episode
        user_prompt_audio: Path to the user's original audio prompt
        intro_jingle: Optional intro jingle
        outro_jingle: Optional outro jingle

    Returns:
        Path to the final episode
    """
    print("Assembling final episode...")

    # Order: intro -> user prompt -> AI dialogue -> outro
    audio_files = []
    if intro_jingle and intro_jingle.exists():
        audio_files.append(intro_jingle)
    if user_prompt_audio and user_prompt_audio.exists():
        audio_files.append(user_prompt_audio)
    audio_files.append(dialogue_audio)
    if outro_jingle and outro_jingle.exists():
        audio_files.append(outro_jingle)

    if len(audio_files) == 1:
        # No jingles, just convert to MP3
        cmd = [
            "ffmpeg", "-y", "-i", str(dialogue_audio),
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    # Normalize all audio files to same format, then concatenate
    temp_dir = output_path.parent / "_temp_concat"
    temp_dir.mkdir(exist_ok=True)

    normalized_files = []
    for i, audio_file in enumerate(audio_files):
        normalized_path = temp_dir / f"temp_normalized_{i}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_file),
            "-ar", "44100", "-ac", "1", "-sample_fmt", "s16",
            str(normalized_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        normalized_files.append(normalized_path)

    # Create file list for concat
    filelist_path = temp_dir / "filelist.txt"
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

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"Final episode saved to: {output_path}")
    return output_path


def generate_episode_metadata(client: genai.Client, script: str) -> dict:
    """
    Generate episode title and description from the script using Gemini.

    Args:
        client: Gemini client
        script: The full podcast script

    Returns:
        Dict with 'title', 'description', and 'image_prompt' keys
    """
    print("Generating episode title, description, and image prompt...")

    metadata_prompt = """Based on this podcast script, generate:

1. A catchy, engaging episode title (max 60 characters)
2. A compelling episode description for podcast platforms (2-3 sentences, ~150-200 words)
3. An image generation prompt for episode artwork (describe a visually striking image that represents the topic, suitable for podcast cover art - abstract, colorful, modern style)

The description should:
- Hook potential listeners with the main topic
- Highlight key insights or surprises discussed
- Use natural, engaging language

Output format (use exactly these labels):
TITLE: [your title here]
DESCRIPTION: [your description here]
IMAGE_PROMPT: [your image prompt here]

Script:
""" + script[:8000]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=metadata_prompt,
    )

    result_text = response.text

    # Parse the response
    title = ""
    description = ""
    image_prompt = ""

    if "TITLE:" in result_text:
        title_start = result_text.index("TITLE:") + len("TITLE:")
        title_end = result_text.index("DESCRIPTION:") if "DESCRIPTION:" in result_text else len(result_text)
        title = result_text[title_start:title_end].strip()

    if "DESCRIPTION:" in result_text:
        desc_start = result_text.index("DESCRIPTION:") + len("DESCRIPTION:")
        desc_end = result_text.index("IMAGE_PROMPT:") if "IMAGE_PROMPT:" in result_text else len(result_text)
        description = result_text[desc_start:desc_end].strip()

    if "IMAGE_PROMPT:" in result_text:
        img_start = result_text.index("IMAGE_PROMPT:") + len("IMAGE_PROMPT:")
        image_prompt = result_text[img_start:].strip()

    return {
        'title': title,
        'description': description,
        'image_prompt': image_prompt
    }


def generate_episode_art(client: genai.Client, image_prompt: str, output_path: Path) -> Path:
    """
    Generate episode artwork using Gemini's image generation.

    Args:
        client: Gemini client
        image_prompt: Description of the image to generate
        output_path: Where to save the image

    Returns:
        Path to the generated image
    """
    print("Generating episode artwork with Gemini...")

    # Enhance the prompt for podcast cover art style
    full_prompt = f"Create podcast episode cover art: {image_prompt}. Style: Modern, bold, visually striking, suitable for a square podcast thumbnail. No text in the image."

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["image", "text"],
        ),
    )

    # Extract and save the image
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            image_data = part.inline_data.data
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"Episode art saved to: {output_path}")
            return output_path

    print("Warning: No image generated")
    return None


def save_metadata_files(metadata: dict, episode_dir: Path):
    """
    Save metadata in both JSON and plain text formats.

    Args:
        metadata: Dict containing episode metadata
        episode_dir: Directory to save the files
    """
    # Save JSON format
    json_path = episode_dir / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save human-readable text format for easy copying
    txt_path = episode_dir / "metadata.txt"
    txt_content = f"""EPISODE METADATA
================

Title:
{metadata.get('title', 'N/A')}

Description:
{metadata.get('description', 'N/A')}

---

Episode Name: {metadata.get('episode_name', 'N/A')}
Generated: {metadata.get('generated_at', 'N/A')}
TTS Engine: {metadata.get('tts_engine', 'N/A')}
Dialogue Segments: {metadata.get('segments_count', 'N/A')}

Voices:
"""
    voices = metadata.get('voices', {})
    for host, voice in voices.items():
        txt_content += f"  - {host}: {voice}\n"

    txt_content += f"""
Files:
  - Audio: {Path(metadata.get('audio_file', '')).name}
  - Script: {Path(metadata.get('script_file', '')).name}
  - Artwork: {Path(metadata.get('artwork_file', '')).name if metadata.get('artwork_file') else 'N/A'}
"""

    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(f"Metadata saved to: {json_path}")
    print(f"Metadata saved to: {txt_path}")


def generate_podcast_episode(
    prompt_audio_path: Path,
    episode_name: str = None,
) -> Path:
    """
    Generate a complete podcast episode from a user's audio prompt using Gemini TTS.

    Workflow:
    1. Send audio prompt to Gemini to generate diarized dialogue script
    2. Parse script into speaker segments
    3. Generate multi-speaker audio using Gemini's native TTS
    4. Add intro/outro jingles

    Args:
        prompt_audio_path: Path to the user's recorded prompt
        episode_name: Optional name for the episode

    Returns:
        Path to the final episode MP3
    """
    if episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{timestamp}"

    # Create episode folder
    episode_dir = EPISODES_DIR / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating podcast episode: {episode_name}")
    print(f"Using Gemini Native Multi-Speaker TTS")
    print(f"Hosts: {HOST_NAME} ({SPEAKER_VOICE_MAP[HOST_NAME]}) & {CO_HOST_NAME} ({SPEAKER_VOICE_MAP[CO_HOST_NAME]})")
    print(f"Output folder: {episode_dir}")
    print(f"{'='*60}\n")

    # Initialize Gemini client
    client = get_gemini_client()

    # Step 1: Transcribe and generate diarized script
    print("Step 1: Generating diarized dialogue script with Gemini...")
    script = transcribe_and_generate_script(client, prompt_audio_path)

    # Save the script
    script_path = episode_dir / "script.txt"
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Script saved to: {script_path}")

    # Step 2: Parse the diarized script
    print("\nStep 2: Parsing diarized script...")
    segments = parse_diarized_script(script)

    if not segments:
        raise ValueError("Failed to parse any dialogue segments from the script")

    # Save segments for reference
    segments_path = episode_dir / "segments.json"
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)

    # Step 3: Generate multi-speaker audio with Gemini TTS
    print(f"\nStep 3: Generating multi-speaker audio with Gemini TTS...")
    dialogue_audio_path = episode_dir / "dialogue.wav"
    generate_multispeaker_audio(client, segments, dialogue_audio_path)

    # Step 4: Assemble final episode (intro + user prompt + dialogue + outro)
    print("\nStep 4: Assembling final episode...")
    episode_path = episode_dir / f"{episode_name}.mp3"
    intro_jingle = JINGLES_DIR / "mixed-intro.mp3"
    outro_jingle = JINGLES_DIR / "mixed-outro.mp3"

    concatenate_episode(
        dialogue_audio=dialogue_audio_path,
        output_path=episode_path,
        user_prompt_audio=prompt_audio_path,
        intro_jingle=intro_jingle if intro_jingle.exists() else None,
        outro_jingle=outro_jingle if outro_jingle.exists() else None,
    )

    # Clean up intermediate dialogue WAV
    if dialogue_audio_path.exists():
        dialogue_audio_path.unlink()

    # Step 5: Generate metadata (title, description, image prompt)
    print("\nStep 5: Generating episode metadata...")
    metadata = generate_episode_metadata(client, script)

    # Step 6: Generate episode artwork
    print("\nStep 6: Generating episode artwork...")
    artwork_path = episode_dir / "artwork.png"
    artwork_file = None
    if metadata.get('image_prompt'):
        try:
            artwork_file = generate_episode_art(client, metadata['image_prompt'], artwork_path)
        except Exception as e:
            print(f"Warning: Failed to generate artwork: {e}")

    # Build full metadata dict
    full_metadata = {
        'title': metadata['title'],
        'description': metadata['description'],
        'episode_name': episode_name,
        'audio_file': str(episode_path),
        'script_file': str(script_path),
        'artwork_file': str(artwork_file) if artwork_file else None,
        'segments_count': len(segments),
        'tts_engine': 'gemini-2.5-pro-preview-tts',
        'voices': {HOST_NAME: SPEAKER_VOICE_MAP[HOST_NAME], CO_HOST_NAME: SPEAKER_VOICE_MAP[CO_HOST_NAME]},
        'generated_at': datetime.now().isoformat(),
    }

    # Save metadata in both JSON and text formats
    save_metadata_files(full_metadata, episode_dir)

    print(f"\n{'='*60}")
    print(f"Episode generated successfully!")
    print(f"{'='*60}")
    print(f"\nTITLE:")
    print(f"  {metadata['title']}")
    print(f"\nDESCRIPTION:")
    print(f"  {metadata['description']}")
    print(f"\nEPISODE FOLDER: {episode_dir}")
    print(f"  - {episode_path.name}")
    print(f"  - script.txt")
    print(f"  - segments.json")
    print(f"  - metadata.json")
    print(f"  - metadata.txt")
    if artwork_file:
        print(f"  - artwork.png")
    print(f"  ({len(segments)} dialogue turns)")
    print(f"{'='*60}\n")

    return episode_path


def process_queue():
    """Process all audio files in the to-process queue."""
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    to_process = [
        f for f in PROMPTS_TO_PROCESS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not to_process:
        print("No audio files found in the to-process queue.")
        print(f"Add audio files to: {PROMPTS_TO_PROCESS_DIR}")
        return

    print(f"Found {len(to_process)} file(s) to process:")
    for f in to_process:
        print(f"  - {f.name}")
    print()

    for prompt_path in to_process:
        try:
            episode_name = prompt_path.stem
            episode_path = generate_podcast_episode(prompt_path, episode_name)

            done_path = PROMPTS_DONE_DIR / prompt_path.name
            prompt_path.rename(done_path)
            print(f"Moved {prompt_path.name} to done folder")

        except Exception as e:
            print(f"Error processing {prompt_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        prompt_path = Path(sys.argv[1])
        if not prompt_path.exists():
            print(f"Error: Audio file not found: {prompt_path}")
            sys.exit(1)
        episode_path = generate_podcast_episode(prompt_path)
        print(f"Done! Episode saved to: {episode_path}")
    else:
        process_queue()


if __name__ == "__main__":
    main()

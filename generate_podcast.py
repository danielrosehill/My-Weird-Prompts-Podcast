#!/usr/bin/env python3
"""
AI Podcast Generator

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to Gemini to transcribe and generate a diarized podcast dialogue script
3. Converts script to multi-speaker audio via Resemble AI TTS (Corn & Herman voices)
4. Concatenates: intro jingle + user audio + AI dialogue + outro jingle

Requires:
    pip install google-genai python-dotenv requests

Environment:
    GEMINI_API_KEY - Your Gemini API key (can be in .env file)
    RESEMBLE_API_KEY - Your Resemble AI API key (can be in .env file)
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
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
RESPONSES_DIR = OUTPUT_DIR / "responses"
EPISODES_DIR = OUTPUT_DIR / "episodes"
JINGLES_DIR = PROJECT_ROOT / "show-elements" / "mixed"

# Ensure output directories exist
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
EPISODES_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DONE_DIR.mkdir(parents=True, exist_ok=True)

# Podcast configuration
PODCAST_NAME = "AI Conversations"
HOST_NAME = "Corn"
CO_HOST_NAME = "Herman"

# Resemble AI Voice UUIDs
CORN_VOICE_UUID = "16048bad"
HERMAN_VOICE_UUID = "efab2c82"

# Resemble AI API configuration
RESEMBLE_API_URL = "https://f.cluster.resemble.ai/synthesize"

# System prompt for generating a diarized podcast dialogue script (~15 minutes)
PODCAST_SCRIPT_PROMPT = """You are a podcast script writer creating an engaging two-host dialogue for "{podcast_name}".

The user has recorded an audio prompt with a topic/question. Listen carefully and generate a ~15 minute podcast episode script as a natural conversation between two AI hosts.

## The Hosts

- **{host_name}**: The curious, enthusiastic host who asks probing questions, shares relatable examples, and keeps the conversation accessible. Tends to think out loud and make connections to everyday life.

- **{co_host_name}**: The knowledgeable expert who provides deep insights, technical details, and authoritative explanations. Balances depth with clarity, using analogies to explain complex topics.

## Script Format

You MUST output the script in this exact diarized format - each line starting with the speaker name followed by a colon:

{host_name}: [dialogue]
{co_host_name}: [dialogue]
{host_name}: [dialogue]
...

## Episode Structure (~15 minutes total when spoken)

1. **Opening Hook** (30 seconds)
   - {host_name} introduces the topic with an intriguing angle
   - {co_host_name} adds a surprising fact or stakes

2. **Topic Introduction** (2 minutes)
   - Both hosts establish what they'll cover
   - Set up why listeners should care

3. **Core Discussion** (8-10 minutes)
   - Back-and-forth exploration of the topic
   - {co_host_name} provides expert insights
   - {host_name} asks clarifying questions, plays devil's advocate
   - Include specific examples, data, case studies
   - Natural tangents that add value

4. **Practical Takeaways** (2-3 minutes)
   - What can listeners actually do with this information?
   - Real-world applications

5. **Closing Thoughts** (1-2 minutes)
   - Future implications
   - Tease what's coming next
   - Sign off

## Dialogue Guidelines

- **Natural speech patterns**: Include filler words occasionally ("you know", "I mean", "right"), false starts, and interruptions
- **Reactions**: "That's fascinating!", "Wait, really?", "Hmm, that's a good point"
- **Length variety**: Mix short reactive lines with longer explanatory passages
- **Chemistry**: The hosts should build on each other's points, occasionally disagree respectfully
- **Engagement hooks**: "Here's the thing...", "What most people don't realize...", "This is where it gets interesting..."

## Content Requirements

- **Depth**: Provide substantive, educational content - not surface-level
- **Specificity**: Use real numbers, names, examples when possible
- **Accuracy**: Be precise on technical topics. Mark speculation clearly.
- **Accessibility**: Explain jargon when used

## Output

Generate ONLY the diarized script. No stage directions, no [brackets], no metadata - just speaker names and their dialogue.

Example format:
{host_name}: Welcome back to {podcast_name}! Today we're diving into something that's been all over the headlines.
{co_host_name}: Yeah, and honestly, most of the coverage has been missing the real story here.
{host_name}: Okay, so break it down for us. What's actually going on?

Now generate the full ~15 minute episode script based on the user's audio prompt.
""".format(podcast_name=PODCAST_NAME, host_name=HOST_NAME, co_host_name=CO_HOST_NAME)


def get_gemini_client() -> genai.Client:
    """Initialize Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Add it to .env file.")
    return genai.Client(api_key=api_key)


def get_resemble_api_key() -> str:
    """Get Resemble AI API key."""
    # Try both common env var names
    api_key = os.environ.get("RESEMBLE_API_KEY") or os.environ.get("RESEMBLE_API")
    if not api_key:
        raise ValueError("RESEMBLE_API_KEY or RESEMBLE_API environment variable not set. Add it to .env file.")
    return api_key


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
    print(f"Generated script:\n{'-'*40}\n{script[:2000]}...\n{'-'*40}")

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
                'voice_uuid': CORN_VOICE_UUID if speaker == HOST_NAME else HERMAN_VOICE_UUID
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
                        'voice_uuid': CORN_VOICE_UUID
                    })
            elif line.startswith(f"{CO_HOST_NAME}:"):
                text = line[len(f"{CO_HOST_NAME}:"):].strip()
                if text:
                    segments.append({
                        'speaker': CO_HOST_NAME,
                        'text': text,
                        'voice_uuid': HERMAN_VOICE_UUID
                    })

    print(f"Parsed {len(segments)} dialogue segments")
    return segments


def synthesize_with_resemble(text: str, voice_uuid: str, output_path: Path) -> Path:
    """
    Synthesize speech using Resemble AI API.

    Args:
        text: Text to synthesize
        voice_uuid: Resemble AI voice UUID
        output_path: Where to save the audio file

    Returns:
        Path to the generated audio file
    """
    api_key = get_resemble_api_key()

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'voice_uuid': voice_uuid,
        'data': text,
        'sample_rate': 44100,
        'output_format': 'wav',
    }

    response = requests.post(RESEMBLE_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Resemble API error: {response.status_code} - {response.text}")

    result = response.json()

    if not result.get('success', False):
        raise Exception(f"Resemble synthesis failed: {result.get('issues', 'Unknown error')}")

    # Decode base64 audio content
    audio_content = base64.b64decode(result['audio_content'])

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(audio_content)

    return output_path


def generate_dialogue_audio(segments: list[dict], output_dir: Path) -> list[Path]:
    """
    Generate audio for all dialogue segments using Resemble AI.

    Args:
        segments: List of parsed dialogue segments
        output_dir: Directory to save audio files

    Returns:
        List of paths to generated audio files in order
    """
    audio_files = []
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(segments)
    for i, segment in enumerate(segments):
        print(f"Synthesizing segment {i+1}/{total} ({segment['speaker']})...")

        output_path = output_dir / f"segment_{i:04d}_{segment['speaker'].lower()}.wav"

        try:
            synthesize_with_resemble(
                text=segment['text'],
                voice_uuid=segment['voice_uuid'],
                output_path=output_path
            )
            audio_files.append(output_path)
        except Exception as e:
            print(f"  Error synthesizing segment {i+1}: {e}")
            # Continue with remaining segments
            continue

    print(f"Generated {len(audio_files)} audio segments")
    return audio_files


def concatenate_audio_segments(
    output_path: Path,
    dialogue_segments: list[Path],
    intro_jingle: Path = None,
    outro_jingle: Path = None,
) -> Path:
    """
    Concatenate dialogue audio segments into final podcast episode using ffmpeg.

    Order: intro (optional) -> dialogue segments -> outro (optional)

    Args:
        output_path: Where to save the final episode
        dialogue_segments: List of paths to dialogue audio files in order
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
    audio_files.extend(dialogue_segments)
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


def cleanup_segment_files(segment_dir: Path):
    """Clean up temporary segment audio files."""
    if segment_dir.exists():
        for f in segment_dir.glob("segment_*.wav"):
            try:
                f.unlink()
            except Exception:
                pass


def generate_episode_metadata(client: genai.Client, script: str) -> dict:
    """
    Generate episode title and description from the script using Gemini.

    Args:
        client: Gemini client
        script: The full podcast script

    Returns:
        Dict with 'title' and 'description' keys
    """
    print("Generating episode title and description...")

    metadata_prompt = """Based on this podcast script, generate:

1. A catchy, engaging episode title (max 60 characters)
2. A compelling episode description for podcast platforms (2-3 sentences, ~150-200 words)

The description should:
- Hook potential listeners with the main topic
- Highlight key insights or surprises discussed
- Use natural, engaging language

Output format (use exactly these labels):
TITLE: [your title here]
DESCRIPTION: [your description here]

Script:
""" + script[:8000]  # Use first 8000 chars to stay within limits

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=metadata_prompt,
    )

    result_text = response.text

    # Parse the response
    title = ""
    description = ""

    if "TITLE:" in result_text:
        title_start = result_text.index("TITLE:") + len("TITLE:")
        title_end = result_text.index("DESCRIPTION:") if "DESCRIPTION:" in result_text else len(result_text)
        title = result_text[title_start:title_end].strip()

    if "DESCRIPTION:" in result_text:
        desc_start = result_text.index("DESCRIPTION:") + len("DESCRIPTION:")
        description = result_text[desc_start:].strip()

    return {
        'title': title,
        'description': description
    }


def generate_podcast_episode(
    prompt_audio_path: Path,
    episode_name: str = None,
) -> Path:
    """
    Generate a complete podcast episode from a user's audio prompt.

    Workflow:
    1. Send audio prompt to Gemini to generate diarized dialogue script
    2. Parse script into speaker segments
    3. Generate audio for each segment using Resemble AI (Corn & Herman voices)
    4. Concatenate all segments with intro/outro jingles

    Args:
        prompt_audio_path: Path to the user's recorded prompt
        episode_name: Optional name for the episode

    Returns:
        Path to the final episode MP3
    """
    if episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{timestamp}"

    # Create episode folder structure: output/episodes/episode_name/
    episode_dir = EPISODES_DIR / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory for segments (will be deleted after concatenation)
    temp_segments_dir = episode_dir / "_temp_segments"

    print(f"\n{'='*60}")
    print(f"Generating podcast episode: {episode_name}")
    print(f"Hosts: {HOST_NAME} (Corn voice) & {CO_HOST_NAME} (Herman voice)")
    print(f"Output folder: {episode_dir}")
    print(f"{'='*60}\n")

    # Initialize Gemini client
    client = get_gemini_client()

    # Step 1: Transcribe and generate diarized script
    print("Step 1: Generating diarized dialogue script with Gemini...")
    script = transcribe_and_generate_script(client, prompt_audio_path)

    # Save the script for reference
    script_path = episode_dir / "script.txt"
    with open(script_path, "w") as f:
        f.write(script)
    print(f"Script saved to: {script_path}")

    # Step 2: Parse the diarized script into segments
    print("\nStep 2: Parsing diarized script...")
    segments = parse_diarized_script(script)

    if not segments:
        raise ValueError("Failed to parse any dialogue segments from the script")

    # Step 3: Generate audio for each segment using Resemble AI
    print(f"\nStep 3: Generating audio for {len(segments)} segments with Resemble AI...")
    dialogue_audio_files = generate_dialogue_audio(segments, temp_segments_dir)

    if not dialogue_audio_files:
        raise ValueError("Failed to generate any audio segments")

    # Step 4: Concatenate into final episode
    print("\nStep 4: Concatenating audio segments...")
    episode_path = episode_dir / f"{episode_name}.mp3"
    intro_jingle = JINGLES_DIR / "mixed-intro.mp3"
    outro_jingle = JINGLES_DIR / "mixed-outro.mp3"

    concatenate_audio_segments(
        output_path=episode_path,
        dialogue_segments=dialogue_audio_files,
        intro_jingle=intro_jingle if intro_jingle.exists() else None,
        outro_jingle=outro_jingle if outro_jingle.exists() else None,
    )

    # Cleanup segment files - delete temp folder and all segments
    print("Cleaning up temporary segment files...")
    if temp_segments_dir.exists():
        shutil.rmtree(temp_segments_dir)

    # Step 5: Generate episode metadata (title and description)
    print("\nStep 5: Generating episode title and description...")
    metadata = generate_episode_metadata(client, script)

    # Save metadata to JSON file
    metadata_path = episode_dir / "metadata.json"
    full_metadata = {
        'title': metadata['title'],
        'description': metadata['description'],
        'episode_name': episode_name,
        'audio_file': str(episode_path),
        'script_file': str(script_path),
        'segments_count': len(segments),
        'generated_at': datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Episode generated successfully!")
    print(f"{'='*60}")
    print(f"\nSUGGESTED TITLE:")
    print(f"  {metadata['title']}")
    print(f"\nSUGGESTED DESCRIPTION:")
    print(f"  {metadata['description']}")
    print(f"\nEPISODE FOLDER: {episode_dir}")
    print(f"  - {episode_path.name}")
    print(f"  - script.txt")
    print(f"  - metadata.json")
    print(f"  ({len(segments)} dialogue turns)")
    print(f"{'='*60}\n")

    return episode_path


def process_queue():
    """Process all audio files in the to-process queue."""
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    # Find all audio files in the to-process directory
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
            # Generate episode using the filename (without extension) as the episode name
            episode_name = prompt_path.stem
            episode_path = generate_podcast_episode(prompt_path, episode_name)

            # Move processed file to done folder
            done_path = PROMPTS_DONE_DIR / prompt_path.name
            prompt_path.rename(done_path)
            print(f"Moved {prompt_path.name} to done folder")

        except Exception as e:
            print(f"Error processing {prompt_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    # If a specific file is provided, process just that file
    if len(sys.argv) > 1:
        prompt_path = Path(sys.argv[1])
        if not prompt_path.exists():
            print(f"Error: Audio file not found: {prompt_path}")
            sys.exit(1)
        episode_path = generate_podcast_episode(prompt_path)
        print(f"Done! Episode saved to: {episode_path}")
    else:
        # Process the queue
        process_queue()


if __name__ == "__main__":
    main()

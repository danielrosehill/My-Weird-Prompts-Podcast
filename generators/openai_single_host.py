#!/usr/bin/env python3
"""
AI Podcast Generator (OpenAI Version - Single Host Response)

A simpler format: Human prompt + single AI host response.
Uses OpenAI for transcription/response generation and cheap TTS.

Episode Format: intro jingle -> original prompt audio -> AI response -> outro jingle

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to OpenAI (via OpenRouter) for transcription and response generation
3. Converts AI response to speech using edge-tts (free) or OpenAI TTS (cheap)
4. Assembles: intro + original prompt + AI response + outro

Requires:
    pip install openai python-dotenv edge-tts

Environment:
    OPENROUTER_API_KEY - Your OpenRouter API key (can be in .env file)
    OPENAI_API_KEY - Optional, for OpenAI TTS (otherwise uses free edge-tts)
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent

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
HOST_NAME = "Nova"  # Single AI host name

# TTS Configuration
# Options: "edge-tts" (free), "openai" (cheap ~$0.015/1K chars)
DEFAULT_TTS_ENGINE = "edge-tts"

# Edge-TTS voice (free Microsoft voices)
# Good options: en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural, en-AU-NatashaNeural
EDGE_TTS_VOICE = "en-US-AriaNeural"

# OpenAI TTS voice (if using OpenAI)
# Options: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE = "nova"
OPENAI_TTS_MODEL = "tts-1"  # tts-1 is cheaper, tts-1-hd is higher quality

# System prompt for generating a podcast-style response
RESPONSE_PROMPT = """You are {host_name}, the friendly and knowledgeable AI host of "{podcast_name}".

The user has recorded an audio message with a topic or question. Listen carefully and provide a comprehensive, engaging response as if you're speaking directly to podcast listeners.

## Your Personality
- Warm, approachable, and genuinely curious
- Knowledgeable but not condescending
- Uses natural speech patterns with occasional "you know", "I mean", "honestly"
- Balances depth with accessibility

## Response Guidelines

1. **Opening Hook** (1-2 sentences)
   - Acknowledge the question/topic enthusiastically
   - Set up why this is interesting or important

2. **Main Content** (the bulk of your response)
   - Provide substantive, educational content
   - Use specific examples, data, and analogies
   - Break down complex topics into digestible parts
   - Address multiple angles or perspectives

3. **Practical Takeaways** (if applicable)
   - What can listeners actually do with this information?
   - Real-world applications

4. **Closing** (1-2 sentences)
   - Summarize the key insight
   - Optionally tease related topics worth exploring

## Speech Style
- Natural, conversational tone (this will be read aloud)
- Vary sentence length for rhythm
- Use engagement hooks: "Here's the thing...", "What's fascinating is...", "The key insight here..."
- Include thoughtful pauses with "..." where appropriate

## Content Requirements
- Be accurate and precise on technical topics
- Mark speculation clearly ("from what we know", "current research suggests")
- Explain jargon when used
- Aim for 2-4 minutes of spoken content (roughly 300-600 words)

Respond ONLY with your spoken response. No stage directions, no [brackets], no meta-commentary.
""".format(host_name=HOST_NAME, podcast_name=PODCAST_NAME)


def get_openrouter_client() -> OpenAI:
    """Initialize OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set. Add it to .env file.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client for TTS."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Add it to .env file.")
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe audio using OpenAI Whisper via OpenRouter.

    Falls back to local whisper if available.
    """
    print(f"Transcribing audio: {audio_path}")

    # Use OpenAI's whisper API
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text

    # Fallback: use local whisper if installed
    try:
        result = subprocess.run(
            ["whisper", str(audio_path), "--model", "base", "--output_format", "txt", "--output_dir", "/tmp"],
            capture_output=True,
            text=True,
            check=True,
        )
        txt_path = Path("/tmp") / f"{audio_path.stem}.txt"
        if txt_path.exists():
            transcript = txt_path.read_text().strip()
            txt_path.unlink()
            return transcript
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise ValueError("No transcription method available. Set OPENAI_API_KEY or install local whisper.")


def generate_response(client: OpenAI, prompt_text: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Generate a podcast-style response to the user's prompt.

    Args:
        client: OpenRouter client
        prompt_text: Transcribed user prompt
        model: Model to use (default: gpt-4o-mini for cost efficiency)

    Returns:
        Generated response text
    """
    print("Generating AI response...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RESPONSE_PROMPT},
            {"role": "user", "content": f"Here's what the listener asked about:\n\n{prompt_text}"},
        ],
        max_tokens=2000,
        temperature=0.8,
    )

    response_text = response.choices[0].message.content
    word_count = len(response_text.split())
    print(f"Generated response with ~{word_count} words")

    return response_text


async def synthesize_with_edge_tts(text: str, output_path: Path, voice: str = EDGE_TTS_VOICE) -> Path:
    """
    Synthesize speech using edge-tts (free Microsoft TTS).

    Args:
        text: Text to synthesize
        output_path: Where to save the audio
        voice: Edge TTS voice name

    Returns:
        Path to generated audio
    """
    import edge_tts

    print(f"Synthesizing with edge-tts ({voice})...")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))

    print(f"Audio saved to: {output_path}")
    return output_path


def synthesize_with_openai(text: str, output_path: Path, voice: str = OPENAI_TTS_VOICE) -> Path:
    """
    Synthesize speech using OpenAI TTS.

    Args:
        text: Text to synthesize
        output_path: Where to save the audio
        voice: OpenAI TTS voice name

    Returns:
        Path to generated audio
    """
    client = get_openai_client()

    print(f"Synthesizing with OpenAI TTS ({voice})...")

    response = client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=voice,
        input=text,
    )

    response.stream_to_file(str(output_path))

    print(f"Audio saved to: {output_path}")
    return output_path


def synthesize_response(text: str, output_path: Path, engine: str = DEFAULT_TTS_ENGINE) -> Path:
    """
    Synthesize speech using the specified TTS engine.

    Args:
        text: Text to synthesize
        output_path: Where to save the audio
        engine: TTS engine ("edge-tts" or "openai")

    Returns:
        Path to generated audio
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "edge-tts":
        return asyncio.run(synthesize_with_edge_tts(text, output_path))
    elif engine == "openai":
        return synthesize_with_openai(text, output_path)
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


def concatenate_episode(
    response_audio: Path,
    output_path: Path,
    user_prompt_audio: Path = None,
    intro_jingle: Path = None,
    outro_jingle: Path = None,
) -> Path:
    """
    Concatenate episode: intro + user prompt + AI response + outro.

    Args:
        response_audio: Path to the AI response audio
        output_path: Where to save the final episode
        user_prompt_audio: Path to the user's original audio prompt
        intro_jingle: Optional intro jingle
        outro_jingle: Optional outro jingle

    Returns:
        Path to the final episode
    """
    print("Assembling final episode...")
    print("  Format: intro -> prompt (original) -> response -> outro")

    # Order: intro -> user prompt -> AI response -> outro
    audio_files = []
    if intro_jingle and intro_jingle.exists():
        audio_files.append(intro_jingle)
        print(f"  + intro: {intro_jingle.name}")
    if user_prompt_audio and user_prompt_audio.exists():
        audio_files.append(user_prompt_audio)
        print(f"  + prompt: {user_prompt_audio.name}")
    audio_files.append(response_audio)
    print(f"  + response: {response_audio.name}")
    if outro_jingle and outro_jingle.exists():
        audio_files.append(outro_jingle)
        print(f"  + outro: {outro_jingle.name}")

    if len(audio_files) == 1:
        # No jingles or prompt, just convert to MP3
        cmd = [
            "ffmpeg", "-y", "-i", str(response_audio),
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


def generate_episode_metadata(client: OpenAI, prompt_text: str, response_text: str) -> dict:
    """
    Generate episode title and description.

    Args:
        client: OpenRouter client
        prompt_text: The user's original prompt
        response_text: The AI's response

    Returns:
        Dict with 'title' and 'description' keys
    """
    print("Generating episode title and description...")

    metadata_prompt = """Based on this Q&A exchange, generate:

1. A catchy, engaging episode title (max 60 characters)
2. A compelling episode description for podcast platforms (2-3 sentences, ~100-150 words)

The description should:
- Hook potential listeners with the main topic
- Highlight key insights discussed
- Use natural, engaging language

Output format (use exactly these labels):
TITLE: [your title here]
DESCRIPTION: [your description here]

USER QUESTION:
{prompt}

AI RESPONSE:
{response}
""".format(prompt=prompt_text[:1000], response=response_text[:2000])

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": metadata_prompt}],
        max_tokens=500,
    )

    result_text = response.choices[0].message.content

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


def save_metadata_files(metadata: dict, episode_dir: Path):
    """Save metadata in both JSON and plain text formats."""
    # Save JSON format
    json_path = episode_dir / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save human-readable text format
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
Host: {metadata.get('host', 'N/A')}
Format: {metadata.get('format', 'N/A')}

Files:
  - Audio: {Path(metadata.get('audio_file', '')).name}
  - Prompt Transcript: {Path(metadata.get('prompt_transcript_file', '')).name}
  - Response Script: {Path(metadata.get('response_script_file', '')).name}
"""

    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(f"Metadata saved to: {json_path}")


def generate_podcast_episode(
    prompt_audio_path: Path,
    episode_name: str = None,
    tts_engine: str = DEFAULT_TTS_ENGINE,
    model: str = "openai/gpt-4o-mini",
) -> Path:
    """
    Generate a complete podcast episode from a user's audio prompt.

    Episode Format: intro -> original prompt audio -> AI response -> outro

    Args:
        prompt_audio_path: Path to the user's recorded prompt
        episode_name: Optional name for the episode
        tts_engine: TTS engine to use ("edge-tts" or "openai")
        model: LLM model for response generation

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
    print(f"Format: intro -> prompt (original audio) -> response -> outro")
    print(f"Host: {HOST_NAME}")
    print(f"TTS Engine: {tts_engine}")
    print(f"LLM: {model}")
    print(f"Output folder: {episode_dir}")
    print(f"{'='*60}\n")

    # Initialize OpenRouter client
    client = get_openrouter_client()

    # Step 1: Transcribe the user's prompt
    print("Step 1: Transcribing user prompt...")
    prompt_text = transcribe_audio(prompt_audio_path)

    # Save the transcript
    prompt_transcript_path = episode_dir / "prompt_transcript.txt"
    with open(prompt_transcript_path, "w") as f:
        f.write(prompt_text)
    print(f"Prompt transcript saved to: {prompt_transcript_path}")
    print(f"Transcript preview: {prompt_text[:200]}...")

    # Step 2: Generate AI response
    print("\nStep 2: Generating AI response...")
    response_text = generate_response(client, prompt_text, model=model)

    # Save the response script
    response_script_path = episode_dir / "response_script.txt"
    with open(response_script_path, "w") as f:
        f.write(response_text)
    print(f"Response script saved to: {response_script_path}")

    # Step 3: Synthesize response audio
    print(f"\nStep 3: Synthesizing response with {tts_engine}...")
    response_audio_path = episode_dir / "response.mp3"
    synthesize_response(response_text, response_audio_path, engine=tts_engine)

    # Step 4: Assemble final episode
    print("\nStep 4: Assembling final episode...")
    episode_path = episode_dir / f"{episode_name}.mp3"
    intro_jingle = JINGLES_DIR / "mixed-intro.mp3"
    outro_jingle = JINGLES_DIR / "mixed-outro.mp3"

    concatenate_episode(
        response_audio=response_audio_path,
        output_path=episode_path,
        user_prompt_audio=prompt_audio_path,
        intro_jingle=intro_jingle if intro_jingle.exists() else None,
        outro_jingle=outro_jingle if outro_jingle.exists() else None,
    )

    # Clean up intermediate response audio
    if response_audio_path.exists():
        response_audio_path.unlink()

    # Step 5: Generate metadata
    print("\nStep 5: Generating episode metadata...")
    metadata = generate_episode_metadata(client, prompt_text, response_text)

    # Build full metadata dict
    full_metadata = {
        'title': metadata['title'],
        'description': metadata['description'],
        'episode_name': episode_name,
        'format': 'intro -> prompt (original audio) -> response -> outro',
        'audio_file': str(episode_path),
        'prompt_transcript_file': str(prompt_transcript_path),
        'response_script_file': str(response_script_path),
        'host': HOST_NAME,
        'tts_engine': tts_engine,
        'llm_model': model,
        'generated_at': datetime.now().isoformat(),
    }

    # Save metadata
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
    print(f"  - prompt_transcript.txt")
    print(f"  - response_script.txt")
    print(f"  - metadata.json / metadata.txt")
    print(f"{'='*60}\n")

    return episode_path


def process_queue(tts_engine: str = DEFAULT_TTS_ENGINE):
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
            episode_path = generate_podcast_episode(prompt_path, episode_name, tts_engine=tts_engine)

            done_path = PROMPTS_DONE_DIR / prompt_path.name
            prompt_path.rename(done_path)
            print(f"Moved {prompt_path.name} to done folder")

        except Exception as e:
            print(f"Error processing {prompt_path.name}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI podcast episode (OpenAI version)")
    parser.add_argument("audio_file", nargs="?", help="Path to audio prompt file")
    parser.add_argument("--tts", choices=["edge-tts", "openai"], default=DEFAULT_TTS_ENGINE,
                        help=f"TTS engine to use (default: {DEFAULT_TTS_ENGINE})")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="LLM model for response generation (default: openai/gpt-4o-mini)")

    args = parser.parse_args()

    if args.audio_file:
        prompt_path = Path(args.audio_file)
        if not prompt_path.exists():
            print(f"Error: Audio file not found: {prompt_path}")
            sys.exit(1)
        episode_path = generate_podcast_episode(prompt_path, tts_engine=args.tts, model=args.model)
        print(f"Done! Episode saved to: {episode_path}")
    else:
        process_queue(tts_engine=args.tts)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AI Podcast Generator (Chatterbox via Replicate - Voice Cloning)

Uses Resemble AI's Chatterbox model on Replicate for instant voice cloning.
Clones voices from sample audio files (Corn & Herman) for natural dialogue.

Cost: ~$0.025 per 1K characters (~$1.88 per 15-min episode)

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to Gemini to transcribe and generate a diarized podcast dialogue script (~15 min)
3. Converts script to multi-speaker audio via Chatterbox (instant voice cloning)
4. Concatenates: intro jingle + user prompt + AI dialogue + outro jingle

Requires:
    pip install google-genai python-dotenv replicate

Environment:
    GEMINI_API_KEY - Your Gemini API key (can be in .env file)
    REPLICATE_API_TOKEN - Your Replicate API token (can be in .env file)
"""

import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import replicate
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Optional: PIL for image handling
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Load environment variables
load_dotenv()

# Configuration
PIPELINE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PIPELINE_ROOT.parent

# Queue-based directory structure
PROMPTS_TO_PROCESS_DIR = PIPELINE_ROOT / "prompts" / "to-process"

# Output directories
OUTPUT_DIR = PIPELINE_ROOT / "output"
EPISODES_DIR = OUTPUT_DIR / "episodes"
JINGLES_DIR = PIPELINE_ROOT / "show-elements" / "mixed"

# Voice samples directory
VOICES_DIR = PROJECT_ROOT / "config" / "voices"

# Ensure output directories exist
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

# Podcast configuration
PODCAST_NAME = "AI Conversations"
HOST_NAME = "Corn"
CO_HOST_NAME = "Herman"

# Voice sample paths for cloning
VOICE_SAMPLES = {
    HOST_NAME: VOICES_DIR / "corn" / "wav" / "corn-1min.wav",
    CO_HOST_NAME: VOICES_DIR / "herman" / "wav" / "herman-1min.wav",
}

# Target episode length (~15 minutes at ~150 words per minute)
TARGET_WORD_COUNT = 2250  # ~15 minutes of dialogue

# Parallel TTS settings
MAX_TTS_WORKERS = 8  # Number of concurrent Replicate API calls (increase for faster generation)

# Audio normalization settings (EBU R128 podcast standard)
TARGET_LUFS = -16  # Podcast standard loudness
TARGET_TP = -1.5   # True peak ceiling

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


def get_replicate_client():
    """Verify Replicate API token is set."""
    api_token = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable not set. Add it to .env file.")
    # Set it for the replicate library
    os.environ["REPLICATE_API_TOKEN"] = api_token
    return True


def transcribe_and_generate_script(client: genai.Client, audio_path: Path) -> str:
    """
    Send audio to Gemini, transcribe it, and generate a diarized podcast dialogue script.
    """
    print(f"Uploading audio file: {audio_path}")

    # Upload the audio file
    audio_file = client.files.upload(file=str(audio_path))
    print(f"Uploaded file: {audio_file.name}")

    # Generate the diarized script
    print("Generating diarized podcast script (~15 min episode)...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PODCAST_SCRIPT_PROMPT, audio_file],
        config=types.GenerateContentConfig(
            max_output_tokens=16000,
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
    """
    segments = []
    pattern = rf'^({HOST_NAME}|{CO_HOST_NAME}):\s*(.+?)(?=^(?:{HOST_NAME}|{CO_HOST_NAME}):|\Z)'

    matches = re.findall(pattern, script, re.MULTILINE | re.DOTALL)

    for speaker, text in matches:
        text = text.strip()
        if text:
            segments.append({
                'speaker': speaker,
                'text': text,
            })

    # Fallback: line-by-line parsing
    if not segments:
        print("Using fallback line-by-line parsing...")
        for line in script.split('\n'):
            line = line.strip()
            if line.startswith(f"{HOST_NAME}:"):
                text = line[len(f"{HOST_NAME}:"):].strip()
                if text:
                    segments.append({'speaker': HOST_NAME, 'text': text})
            elif line.startswith(f"{CO_HOST_NAME}:"):
                text = line[len(f"{CO_HOST_NAME}:"):].strip()
                if text:
                    segments.append({'speaker': CO_HOST_NAME, 'text': text})

    print(f"Parsed {len(segments)} dialogue segments")
    return segments


def upload_voice_samples() -> dict[str, str]:
    """
    Upload voice samples to Replicate once and return reusable URLs.

    Returns:
        Dict mapping speaker name to uploaded file URL
    """
    uploaded_urls = {}

    for speaker, sample_path in VOICE_SAMPLES.items():
        if not sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found for {speaker}: {sample_path}")

        print(f"  Uploading voice sample for {speaker}...")
        file = replicate.files.create(
            file=sample_path,
            metadata={"speaker": speaker, "purpose": "voice_cloning"}
        )
        uploaded_urls[speaker] = file.urls['get']
        print(f"    Uploaded: {file.id}")

    return uploaded_urls


def cleanup_uploaded_files(uploaded_urls: dict[str, str]):
    """
    Delete uploaded voice samples from Replicate after use.
    """
    for speaker, url in uploaded_urls.items():
        try:
            # Extract file ID from URL and delete
            # URLs look like: https://replicate.delivery/pbxt/FILE_ID/filename
            # We need to list files and find by URL
            pass  # Files auto-expire, skip cleanup for now
        except Exception as e:
            print(f"  Warning: Could not clean up file for {speaker}: {e}")


def synthesize_with_chatterbox(text: str, voice_sample_url: str, output_path: Path) -> Path:
    """
    Synthesize speech using Chatterbox on Replicate with instant voice cloning.

    Args:
        text: Text to synthesize
        voice_sample_url: URL to pre-uploaded voice sample (from replicate.files.create)
        output_path: Where to save the generated audio

    Returns:
        Path to the generated audio file
    """
    # Run Chatterbox on Replicate using pre-uploaded URL
    output = replicate.run(
        "resemble-ai/chatterbox:1b8422bc49635c20d0a84e387ed20879c0dd09254ecdb4e75dc4bec10ff94e97",
        input={
            "prompt": text,
            "audio_prompt": voice_sample_url,  # Use URL instead of file object
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
        }
    )

    # Download the output - handle both URL string and FileOutput object
    import urllib.request
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the URL from the output (handles FileOutput objects from newer replicate lib)
    output_url = str(output) if hasattr(output, '__str__') else output
    if hasattr(output, 'url'):
        output_url = output.url
    elif hasattr(output, 'read'):
        # It's a file-like object, read and write directly
        with open(output_path, 'wb') as f:
            f.write(output.read())
        return output_path

    urllib.request.urlretrieve(output_url, str(output_path))

    return output_path


def synthesize_segment_task(args: tuple) -> tuple[int, Path, Exception | None]:
    """
    Task function for parallel TTS synthesis.

    Args:
        args: Tuple of (index, segment, voice_url, output_path)

    Returns:
        Tuple of (index, output_path, error or None)
    """
    i, segment, voice_url, output_path = args
    try:
        synthesize_with_chatterbox(segment['text'], voice_url, output_path)
        return (i, output_path, None)
    except Exception as e:
        return (i, output_path, e)


def generate_dialogue_audio(segments: list[dict], episode_dir: Path, voice_urls: dict[str, str]) -> Path:
    """
    Generate audio for all dialogue segments using Chatterbox with parallel processing.

    Args:
        segments: List of parsed dialogue segments
        episode_dir: Directory to save intermediate files
        voice_urls: Dict mapping speaker name to pre-uploaded voice sample URL

    Returns:
        Path to the combined dialogue audio
    """
    temp_dir = episode_dir / "_temp_tts"
    temp_dir.mkdir(exist_ok=True)

    total_chars = sum(len(s['text']) for s in segments)
    estimated_cost = (total_chars / 1000) * 0.025
    print(f"Generating {len(segments)} segments (~{total_chars} chars, est. cost: ${estimated_cost:.2f})")
    print(f"Using {MAX_TTS_WORKERS} parallel workers for TTS generation...")

    # Prepare tasks for parallel execution
    tasks = []
    for i, segment in enumerate(segments):
        speaker = segment['speaker']
        voice_url = voice_urls[speaker]
        segment_path = temp_dir / f"segment_{i:03d}_{speaker.lower()}.mp3"
        tasks.append((i, segment, voice_url, segment_path))

    # Execute TTS in parallel with thread pool
    results = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_TTS_WORKERS) as executor:
        futures = {executor.submit(synthesize_segment_task, task): task[0] for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            idx, output_path, error = future.result()
            completed += 1
            speaker = segments[idx]['speaker']
            text_preview = segments[idx]['text'][:40]

            if error:
                print(f"  [{completed}/{len(segments)}] FAILED {speaker}: {error}")
                raise error
            else:
                print(f"  [{completed}/{len(segments)}] {speaker}: {text_preview}...")
                results[idx] = output_path

    # Sort results by original index to maintain dialogue order
    segment_files = [results[i] for i in sorted(results.keys())]

    # Concatenate all segments
    print(f"Concatenating {len(segment_files)} audio segments...")

    filelist_path = temp_dir / "filelist.txt"
    with open(filelist_path, "w") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    dialogue_path = episode_dir / "dialogue.mp3"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100",
        str(dialogue_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Cleanup temp files
    shutil.rmtree(temp_dir)

    return dialogue_path


def normalize_audio_loudness(input_path: Path, output_path: Path, target_lufs: float = TARGET_LUFS) -> Path:
    """
    Normalize audio to target loudness using EBU R128 two-pass loudnorm filter.

    Args:
        input_path: Input audio file
        output_path: Output normalized audio file
        target_lufs: Target integrated loudness (default -16 LUFS for podcasts)

    Returns:
        Path to normalized audio
    """
    # First pass: analyze loudness
    analyze_cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP={TARGET_TP}:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(analyze_cmd, capture_output=True, text=True)

    # Parse loudness stats from stderr (ffmpeg outputs to stderr)
    stderr = result.stderr
    try:
        # Find the JSON block in stderr
        json_start = stderr.rfind('{')
        json_end = stderr.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            stats_json = stderr[json_start:json_end]
            stats = json.loads(stats_json)
        else:
            # Fallback to single-pass if we can't parse
            print("    Warning: Could not parse loudness stats, using single-pass normalization")
            stats = None
    except json.JSONDecodeError:
        stats = None

    # Second pass: apply normalization with measured values
    if stats:
        normalize_cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", (
                f"loudnorm=I={target_lufs}:TP={TARGET_TP}:LRA=11:"
                f"measured_I={stats.get('input_i', -24)}:"
                f"measured_TP={stats.get('input_tp', -2)}:"
                f"measured_LRA={stats.get('input_lra', 7)}:"
                f"measured_thresh={stats.get('input_thresh', -34)}:"
                f"offset={stats.get('target_offset', 0)}:"
                f"linear=true:print_format=summary"
            ),
            "-ar", "44100", "-ac", "1",
            str(output_path)
        ]
    else:
        # Single-pass fallback
        normalize_cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"loudnorm=I={target_lufs}:TP={TARGET_TP}:LRA=11",
            "-ar", "44100", "-ac", "1",
            str(output_path)
        ]

    subprocess.run(normalize_cmd, capture_output=True, check=True)
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
    Applies EBU R128 loudness normalization to each component for consistent volume.
    """
    print("Assembling final episode with loudness normalization...")

    audio_files = []
    labels = []  # For logging
    if intro_jingle and intro_jingle.exists():
        audio_files.append(intro_jingle)
        labels.append("intro")
    if user_prompt_audio and user_prompt_audio.exists():
        audio_files.append(user_prompt_audio)
        labels.append("prompt")
    audio_files.append(dialogue_audio)
    labels.append("dialogue")
    if outro_jingle and outro_jingle.exists():
        audio_files.append(outro_jingle)
        labels.append("outro")

    # Create temp directory for normalized files
    temp_dir = output_path.parent / "_temp_concat"
    temp_dir.mkdir(exist_ok=True)

    # Normalize each audio file to podcast standard (-16 LUFS)
    print(f"  Normalizing {len(audio_files)} audio segments to {TARGET_LUFS} LUFS...")
    normalized_files = []
    for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
        print(f"    [{i+1}/{len(audio_files)}] Normalizing {label}...")
        normalized_path = temp_dir / f"normalized_{i}_{label}.wav"
        normalize_audio_loudness(audio_file, normalized_path)
        normalized_files.append(normalized_path)

    # Create file list for concatenation
    filelist_path = temp_dir / "filelist.txt"
    with open(filelist_path, "w") as f:
        for nf in normalized_files:
            f.write(f"file '{nf}'\n")

    # Concatenate normalized files
    print("  Concatenating normalized segments...")
    concat_path = temp_dir / "concatenated.wav"
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c:a", "pcm_s16le",
        str(concat_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Final pass: normalize the complete episode and encode to MP3
    print("  Final loudness pass and MP3 encoding...")
    final_normalize_cmd = [
        "ffmpeg", "-y", "-i", str(concat_path),
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11",
        "-c:a", "libmp3lame", "-b:a", "192k",  # Higher bitrate for quality
        str(output_path)
    ]
    subprocess.run(final_normalize_cmd, capture_output=True, check=True)

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"Final episode saved to: {output_path}")
    return output_path


def generate_episode_metadata(client: genai.Client, script: str) -> dict:
    """Generate episode title and description from the script using Gemini."""
    print("Generating episode title, description, and image prompt...")

    metadata_prompt = """Based on this podcast script, generate:

1. A catchy, engaging episode title (max 60 characters)
2. A compelling episode description for podcast platforms (2-3 sentences, ~150-200 words)
3. An image generation prompt for episode artwork

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


def generate_cover_art(image_prompt: str, episode_dir: Path, num_variants: int = 3) -> list[Path]:
    """
    Generate multiple episode cover art variants using Replicate (Flux Schnell).

    Args:
        image_prompt: Prompt describing the desired cover art
        episode_dir: Episode directory to save images in
        num_variants: Number of cover art variants to generate (default 3)

    Returns:
        List of paths to generated images (may be empty if all failed)
    """
    print(f"Generating {num_variants} cover art variants with Replicate (Flux Schnell)...")

    # Enhance the prompt for podcast cover art style
    # CRITICAL: Explicitly forbid any text elements - AI image generators often produce garbled pseudo-text
    enhanced_prompt = f"""Professional podcast episode cover art, modern clean design, visually striking, suitable for podcast platforms, square format. IMPORTANT: Do NOT include any text, words, letters, numbers, typography, titles, labels, or writing of any kind. No signs, no logos with text, no speech bubbles. Pure visual imagery only - abstract or symbolic representation. Theme: {image_prompt}"""

    generated_paths = []

    try:
        import urllib.request

        # Generate multiple variants in parallel via single API call
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": enhanced_prompt,
                "aspect_ratio": "1:1",
                "output_format": "png",
                "num_outputs": num_variants,
            }
        )

        episode_dir.mkdir(parents=True, exist_ok=True)

        # Handle output (could be a list or single URL)
        if not isinstance(output, list):
            output = [output]

        for i, image_output in enumerate(output):
            try:
                # Get URL from output
                if hasattr(image_output, 'url'):
                    image_url = image_output.url
                else:
                    image_url = str(image_output)

                # Save with numbered suffix (cover_1.png, cover_2.png, cover_3.png)
                output_path = episode_dir / f"cover_{i+1}.png"
                urllib.request.urlretrieve(image_url, str(output_path))
                generated_paths.append(output_path)
                print(f"  Cover art {i+1}/{num_variants} saved: {output_path.name}")
            except Exception as e:
                print(f"  Warning: Failed to save cover art {i+1}: {e}")

    except Exception as e:
        print(f"  Warning: Cover art generation failed: {e}")

    return generated_paths


def save_metadata_files(metadata: dict, episode_dir: Path):
    """Save metadata in both JSON and plain text formats."""
    json_path = episode_dir / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

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

Voices (cloned from samples):
"""
    voices = metadata.get('voice_samples', {})
    for host, sample in voices.items():
        txt_content += f"  - {host}: {sample}\n"

    txt_content += f"""
Files:
  - Audio: {Path(metadata.get('audio_file', '')).name}
  - Script: {Path(metadata.get('script_file', '')).name}
"""

    cover_art = metadata.get('cover_art')
    if cover_art:
        txt_content += "\nCover Art Options:\n"
        if isinstance(cover_art, list):
            for i, ca in enumerate(cover_art, 1):
                txt_content += f"  - Option {i}: {Path(ca).name}\n"
        else:
            txt_content += f"  - {Path(cover_art).name}\n"

    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(f"Metadata saved to: {json_path}")


def generate_podcast_episode(
    prompt_audio_path: Path,
    episode_name: str = None,
) -> Path:
    """
    Generate a complete podcast episode from a user's audio prompt using Chatterbox TTS.

    Optimized workflow with parallel operations:
    1. Upload voice samples + generate script (parallel)
    2. Parse script into segments
    3. Generate metadata + dialogue audio (parallel - metadata doesn't need audio)
    4. Generate cover art (parallel with TTS)
    5. Assemble final episode

    Args:
        prompt_audio_path: Path to the user's recorded prompt
        episode_name: Optional name for the episode

    Returns:
        Path to the final episode MP3
    """
    if episode_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = f"episode_{timestamp}"

    episode_dir = EPISODES_DIR / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating podcast episode: {episode_name}")
    print(f"Using Chatterbox (Replicate) with Voice Cloning")
    print(f"Hosts: {HOST_NAME} & {CO_HOST_NAME}")
    print(f"Cost: ~$0.025 per 1K characters + ~$0.01 for cover art")
    print(f"Output folder: {episode_dir}")
    print(f"{'='*60}\n")

    # Verify API keys and voice samples upfront
    gemini_client = get_gemini_client()
    get_replicate_client()

    for speaker, sample_path in VOICE_SAMPLES.items():
        if not sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found for {speaker}: {sample_path}")
        print(f"Voice sample for {speaker}: {sample_path.name}")

    # OPTIMIZATION: Run voice upload and script generation in parallel
    # Voice upload is slow (network), script gen is slow (LLM) - do both at once
    print("\nStep 1: Uploading voice samples + generating script (parallel)...")

    voice_urls = None
    script = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        voice_future = executor.submit(upload_voice_samples)
        script_future = executor.submit(transcribe_and_generate_script, gemini_client, prompt_audio_path)

        # Wait for both to complete
        voice_urls = voice_future.result()
        print("  Voice samples uploaded")
        script = script_future.result()
        print("  Script generated")

    # Save script
    script_path = episode_dir / "script.txt"
    with open(script_path, "w") as f:
        f.write(script)

    # Step 2: Parse script (fast, no parallelization needed)
    print("\nStep 2: Parsing diarized script...")
    segments = parse_diarized_script(script)

    if not segments:
        raise ValueError("Failed to parse any dialogue segments from the script")

    segments_path = episode_dir / "segments.json"
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)

    # OPTIMIZATION: Generate metadata, cover art, and dialogue audio in parallel
    # Metadata + cover art only need the script, not the audio
    print("\nStep 3: Generating metadata + cover art + dialogue audio (parallel)...")

    metadata = None
    cover_art_paths = []
    dialogue_audio_path = None

    def generate_metadata_and_cover():
        """Generate metadata then cover art (cover needs metadata's image prompt)."""
        nonlocal metadata, cover_art_paths
        metadata = generate_episode_metadata(gemini_client, script)
        if metadata.get('image_prompt'):
            cover_art_paths = generate_cover_art(metadata['image_prompt'], episode_dir, num_variants=3)
        return metadata, cover_art_paths

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Start metadata+cover generation
        metadata_future = executor.submit(generate_metadata_and_cover)
        # Start TTS generation (the heavy lift)
        audio_future = executor.submit(generate_dialogue_audio, segments, episode_dir, voice_urls)

        # Wait for both - audio is usually much slower
        metadata, cover_art_paths = metadata_future.result()
        print("  Metadata and cover art complete")
        dialogue_audio_path = audio_future.result()
        print("  Dialogue audio complete")

    # Step 4: Assemble final episode
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

    # Cleanup dialogue intermediate file
    if dialogue_audio_path.exists():
        dialogue_audio_path.unlink()

    full_metadata = {
        'title': metadata['title'],
        'description': metadata['description'],
        'image_prompt': metadata.get('image_prompt', ''),
        'cover_art': [str(p) for p in cover_art_paths] if cover_art_paths else None,
        'episode_name': episode_name,
        'audio_file': str(episode_path),
        'script_file': str(script_path),
        'segments_count': len(segments),
        'tts_engine': 'chatterbox-replicate',
        'voice_samples': {
            HOST_NAME: str(VOICE_SAMPLES[HOST_NAME]),
            CO_HOST_NAME: str(VOICE_SAMPLES[CO_HOST_NAME]),
        },
        'generated_at': datetime.now().isoformat(),
    }

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
    print(f"  - metadata.json / metadata.txt")
    if cover_art_paths:
        for cap in cover_art_paths:
            print(f"  - {cap.name}")
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

            # Delete the prompt file after successful processing
            prompt_path.unlink()
            print(f"Deleted processed prompt: {prompt_path.name}")

        except Exception as e:
            print(f"Error processing {prompt_path.name}: {e}")
            print(f"  (prompt file kept for retry)")
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

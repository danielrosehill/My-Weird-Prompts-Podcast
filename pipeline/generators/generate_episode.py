#!/usr/bin/env python3
"""
AI Podcast Generator (Chatterbox via fal.ai - Voice Cloning)

Uses Resemble AI's Chatterbox model on fal.ai for instant voice cloning.
Clones voices from sample audio files (Corn & Herman) for natural dialogue.

Workflow:
1. Takes a human-recorded audio prompt
2. Sends to Gemini to transcribe and generate a diarized podcast dialogue script (~15 min)
3. Converts script to multi-speaker audio via Chatterbox (instant voice cloning)
4. Concatenates: intro jingle + user prompt + AI dialogue + outro jingle

Requires:
    pip install google-genai python-dotenv fal-client

Environment:
    GEMINI_API_KEY - Your Gemini API key (can be in .env file)
    FAL_KEY - Your fal.ai API key (can be in .env file as FAL_API_KEY)
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

import fal_client
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

# NAS destination for finished episodes
NAS_DESTINATION = Path("/mnt/nas/AI-Podcasts/My_Weird_Prompts")

# Voice samples directory
VOICES_DIR = PROJECT_ROOT / "config" / "voices"

# Ensure output directories exist
EPISODES_DIR.mkdir(parents=True, exist_ok=True)

# Podcast configuration
PODCAST_NAME = "My Weird Prompts"
PODCAST_SUBTITLE = "A Human-AI Podcast Collaboration"
PRODUCER_NAME = "Daniel Rosehill"
HOST_NAME = "Corn"
CO_HOST_NAME = "Herman"

# Disclaimer audio path
DISCLAIMER_PATH = PIPELINE_ROOT / "show-elements" / "mixed" / "disclaimer.mp3"

# Voice sample paths for cloning
VOICE_SAMPLES = {
    HOST_NAME: VOICES_DIR / "corn" / "wav" / "corn-1min.wav",
    CO_HOST_NAME: VOICES_DIR / "herman" / "wav" / "herman-1min.wav",
}

# Target episode length (~20 minutes at ~150 words per minute)
TARGET_WORD_COUNT = 3000  # ~20 minutes of dialogue

# Parallel TTS settings
MAX_TTS_WORKERS = 4  # Number of concurrent Replicate API calls (keep low to avoid rate limits)

# Audio normalization settings (EBU R128 podcast standard)
TARGET_LUFS = -16  # Podcast standard loudness
TARGET_TP = -1.5   # True peak ceiling

# Prompt audio processing settings
SILENCE_THRESHOLD_DB = -35  # Audio below this level is considered silence
MIN_SILENCE_DURATION = 0.3  # Silence must be at least this long to be detected (seconds)
MAX_SILENCE_DURATION = 0.4  # Compress longer silences down to this duration
TRIM_LEADING_TRAILING = True  # Remove silence at start/end of prompt

# System prompt for generating a diarized podcast dialogue script (~20 minutes)
PODCAST_SCRIPT_PROMPT = """You are a podcast script writer creating an engaging two-host dialogue for "{podcast_name}" ({podcast_subtitle}).

## About This Podcast

"{podcast_name}" is a unique human-AI collaboration podcast produced by {producer_name}. Daniel records audio prompts with topics, questions, or ideas he wants explored, and the AI hosts ({host_name} and {co_host_name}) discuss them in depth. The podcast is available on Spotify and other major podcast platforms.

**IMPORTANT**: Daniel is NOT "a listener" - he is the producer and creator of the show who sends in the prompts. When referring to the prompt, say things like "Daniel wanted us to explore...", "Daniel's asked us to dig into...", or "This week Daniel sent us a fascinating prompt about...". Never say "a listener asked" or similar.

## The Hosts

- **{host_name}**: The curious, enthusiastic host who asks probing questions, shares relatable examples, and keeps the conversation accessible. Tends to think out loud and make connections to everyday life.

- **{co_host_name}**: The knowledgeable expert who provides deep insights, technical details, and authoritative explanations. Balances depth with clarity, using analogies to explain complex topics.

## Script Format

You MUST output the script in this exact diarized format - each line starting with the speaker name followed by a colon:

{host_name}: [dialogue]
{co_host_name}: [dialogue]
{host_name}: [dialogue]
...

## Episode Structure (~20 minutes total when spoken, approximately 2750-3250 words)

1. **Opening Hook** (30 seconds)
   - {host_name} welcomes listeners to {podcast_name} and introduces today's topic
   - Reference that Daniel sent in this prompt
   - {co_host_name} adds a surprising fact or stakes

2. **Topic Introduction** (2 minutes)
   - Both hosts establish what they'll cover
   - Set up why listeners should care

3. **Core Discussion** (12-14 minutes)
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
   - Thank Daniel for the prompt
   - Remind listeners they can find {podcast_name} on Spotify and wherever they get their podcasts
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
- **Length**: AIM FOR 2750-3250 WORDS TOTAL. This is critical for reaching ~20 minutes.

## Output

Generate ONLY the diarized script. No stage directions, no [brackets], no metadata - just speaker names and their dialogue.

Example format:
{host_name}: Welcome to {podcast_name}! I'm {host_name}, and as always I'm here with {co_host_name}. Daniel sent us a really interesting prompt this week - he wants us to dig into something that's been all over the headlines lately.
{co_host_name}: Yeah, and I think what's interesting is that most of the coverage has been missing the real story here. There's this whole dimension that people aren't talking about.
{host_name}: Okay, so break it down for us. What's actually going on beneath the surface?

Now generate the full ~20 minute episode script (2750-3250 words) based on Daniel's audio prompt.
""".format(
    podcast_name=PODCAST_NAME,
    podcast_subtitle=PODCAST_SUBTITLE,
    producer_name=PRODUCER_NAME,
    host_name=HOST_NAME,
    co_host_name=CO_HOST_NAME
)


def get_gemini_client() -> genai.Client:
    """Initialize Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Add it to .env file.")
    return genai.Client(api_key=api_key)


def get_fal_client():
    """Verify fal.ai API key is set."""
    # fal_client looks for FAL_KEY env var
    api_key = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    if not api_key:
        raise ValueError("FAL_KEY environment variable not set. Add it to .env file.")
    # Set it for the fal_client library (it expects FAL_KEY)
    os.environ["FAL_KEY"] = api_key
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
    print("Generating diarized podcast script (~20 min episode)...")
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
    Upload voice samples to fal.ai CDN once and return reusable URLs.

    Returns:
        Dict mapping speaker name to uploaded file URL
    """
    uploaded_urls = {}

    for speaker, sample_path in VOICE_SAMPLES.items():
        if not sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found for {speaker}: {sample_path}")

        print(f"  Uploading voice sample for {speaker}...")
        url = fal_client.upload_file(str(sample_path))
        uploaded_urls[speaker] = url
        print(f"    Uploaded: {url[:60]}...")

    return uploaded_urls


def synthesize_with_chatterbox(text: str, voice_sample_url: str, output_path: Path) -> Path:
    """
    Synthesize speech using Chatterbox on fal.ai with instant voice cloning.

    Args:
        text: Text to synthesize
        voice_sample_url: URL to pre-uploaded voice sample (from fal_client.upload_file)
        output_path: Where to save the generated audio

    Returns:
        Path to the generated audio file
    """
    # Run Chatterbox on fal.ai
    result = fal_client.subscribe(
        "fal-ai/chatterbox/text-to-speech",
        arguments={
            "text": text,
            "audio_url": voice_sample_url,
            "exaggeration": 0.5,
            "cfg": 0.5,
            "temperature": 0.7,
        }
    )

    # Download the output audio
    import urllib.request
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_url = result["audio"]["url"]
    urllib.request.urlretrieve(audio_url, str(output_path))

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


def process_prompt_audio(input_path: Path, output_path: Path) -> Path:
    """
    Process the user's prompt audio to tighten up pacing:
    1. Trim leading/trailing silence
    2. Detect internal silences and compress long pauses
    3. Output a cleaner, more broadcast-ready version

    Args:
        input_path: Original prompt audio file
        output_path: Where to save the processed audio

    Returns:
        Path to processed audio file
    """
    print(f"Processing prompt audio: {input_path.name}")

    temp_dir = output_path.parent / "_temp_prompt_processing"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Detect silences in the audio
        print("  Detecting silences...")
        silence_detect_cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={MIN_SILENCE_DURATION}",
            "-f", "null", "-"
        ]
        result = subprocess.run(silence_detect_cmd, capture_output=True, text=True)
        stderr = result.stderr

        # Parse silence intervals from ffmpeg output
        silence_starts = []
        silence_ends = []
        for line in stderr.split('\n'):
            if 'silence_start:' in line:
                try:
                    start = float(line.split('silence_start:')[1].split()[0])
                    silence_starts.append(start)
                except (ValueError, IndexError):
                    pass
            elif 'silence_end:' in line:
                try:
                    end = float(line.split('silence_end:')[1].split()[0])
                    silence_ends.append(end)
                except (ValueError, IndexError):
                    pass

        # Get audio duration
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        total_duration = float(duration_result.stdout.strip())

        # Pair up silence intervals
        silences = []
        for i, start in enumerate(silence_starts):
            if i < len(silence_ends):
                silences.append((start, silence_ends[i]))
            else:
                # Silence extends to end of file
                silences.append((start, total_duration))

        if not silences:
            print("  No significant silences detected, copying original")
            shutil.copy(input_path, output_path)
            shutil.rmtree(temp_dir)
            return output_path

        print(f"  Found {len(silences)} silence regions")

        # Step 2: Build speech segments (inverse of silences)
        # Also handle leading/trailing silence trimming
        speech_segments = []
        current_pos = 0.0

        # Handle leading silence
        if TRIM_LEADING_TRAILING and silences and silences[0][0] < 0.1:
            # First silence starts at beginning - skip it
            current_pos = silences[0][1]
            silences = silences[1:]
            print(f"  Trimming {current_pos:.2f}s leading silence")

        for silence_start, silence_end in silences:
            # Add speech segment before this silence
            if silence_start > current_pos:
                speech_segments.append({
                    'start': current_pos,
                    'end': silence_start,
                    'type': 'speech'
                })

            # Calculate silence duration and decide how to handle it
            silence_duration = silence_end - silence_start

            if silence_duration > MAX_SILENCE_DURATION:
                # Compress long silence to MAX_SILENCE_DURATION
                speech_segments.append({
                    'start': silence_start,
                    'end': silence_start + MAX_SILENCE_DURATION,
                    'type': 'silence_compressed',
                    'original_duration': silence_duration
                })
            else:
                # Keep short silences as-is
                speech_segments.append({
                    'start': silence_start,
                    'end': silence_end,
                    'type': 'silence_kept'
                })

            current_pos = silence_end

        # Add final speech segment
        if current_pos < total_duration:
            final_segment_end = total_duration

            # Handle trailing silence
            if TRIM_LEADING_TRAILING and silences:
                last_silence_start, last_silence_end = silences[-1] if silences else (0, 0)
                if last_silence_end >= total_duration - 0.1:
                    # Last silence extends to end - already handled by stopping at silence_start
                    pass

            speech_segments.append({
                'start': current_pos,
                'end': final_segment_end,
                'type': 'speech'
            })

        # Handle trailing silence trimming
        if TRIM_LEADING_TRAILING and speech_segments:
            last_seg = speech_segments[-1]
            if last_seg['type'] in ('silence_compressed', 'silence_kept'):
                removed = speech_segments.pop()
                print(f"  Trimming {removed['end'] - removed['start']:.2f}s trailing silence")

        # Step 3: Extract and concatenate segments
        print(f"  Extracting {len(speech_segments)} segments...")
        segment_files = []

        for i, seg in enumerate(speech_segments):
            seg_path = temp_dir / f"seg_{i:03d}.wav"
            duration = seg['end'] - seg['start']

            extract_cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ss", str(seg['start']),
                "-t", str(duration),
                "-c:a", "pcm_s16le", "-ar", "44100",
                str(seg_path)
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)
            segment_files.append(seg_path)

        # Step 4: Concatenate all segments
        print("  Concatenating processed segments...")
        filelist_path = temp_dir / "filelist.txt"
        with open(filelist_path, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        # Concatenate to output format (preserve original format or use wav as intermediate)
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(filelist_path),
            "-c:a", "pcm_s16le", "-ar", "44100",
            str(output_path)
        ]
        subprocess.run(concat_cmd, capture_output=True, check=True)

        # Calculate time saved
        original_duration = total_duration
        new_duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)
        ]
        new_duration_result = subprocess.run(new_duration_cmd, capture_output=True, text=True)
        new_duration = float(new_duration_result.stdout.strip())

        time_saved = original_duration - new_duration
        print(f"  Original: {original_duration:.1f}s -> Processed: {new_duration:.1f}s (saved {time_saved:.1f}s)")

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return output_path


def concatenate_episode(
    dialogue_audio: Path,
    output_path: Path,
    user_prompt_audio: Path = None,
    intro_jingle: Path = None,
    disclaimer_audio: Path = None,
    outro_jingle: Path = None,
) -> Path:
    """
    Concatenate all episode audio: intro + disclaimer + user prompt + dialogue + outro.
    Applies EBU R128 loudness normalization to each component for consistent volume.
    """
    print("Assembling final episode with loudness normalization...")

    audio_files = []
    labels = []  # For logging
    if intro_jingle and intro_jingle.exists():
        audio_files.append(intro_jingle)
        labels.append("intro")
    if disclaimer_audio and disclaimer_audio.exists():
        audio_files.append(disclaimer_audio)
        labels.append("disclaimer")
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
    Generate multiple episode cover art variants using fal.ai (Flux Schnell).

    Args:
        image_prompt: Prompt describing the desired cover art
        episode_dir: Episode directory to save images in
        num_variants: Number of cover art variants to generate (default 3)

    Returns:
        List of paths to generated images (may be empty if all failed)
    """
    print(f"Generating {num_variants} cover art variants with fal.ai (Flux Schnell)...")

    # Enhance the prompt for podcast cover art style
    # CRITICAL: Explicitly forbid any text elements - AI image generators often produce garbled pseudo-text
    enhanced_prompt = f"""Professional podcast episode cover art, modern clean design, visually striking, suitable for podcast platforms, square format. IMPORTANT: Do NOT include any text, words, letters, numbers, typography, titles, labels, or writing of any kind. No signs, no logos with text, no speech bubbles. Pure visual imagery only - abstract or symbolic representation. Theme: {image_prompt}"""

    generated_paths = []

    try:
        import urllib.request

        # Save images to images/ subfolder
        images_dir = episode_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Generate each variant (fal.ai flux-schnell doesn't support num_outputs)
        for i in range(num_variants):
            try:
                result = fal_client.subscribe(
                    "fal-ai/flux/schnell",
                    arguments={
                        "prompt": enhanced_prompt,
                        "image_size": "square",
                        "num_images": 1,
                    }
                )

                # Get URL from result
                image_url = result["images"][0]["url"]

                # Save with numbered suffix (cover_1.png, cover_2.png, cover_3.png)
                output_path = images_dir / f"cover_{i+1}.png"
                urllib.request.urlretrieve(image_url, str(output_path))
                generated_paths.append(output_path)
                print(f"  Cover art {i+1}/{num_variants} saved: images/{output_path.name}")
            except Exception as e:
                print(f"  Warning: Failed to generate cover art {i+1}: {e}")

    except Exception as e:
        print(f"  Warning: Cover art generation failed: {e}")

    return generated_paths


def save_metadata_files(metadata: dict, episode_dir: Path):
    """Save metadata in both JSON and plain text formats to metadata/ subfolder."""
    metadata_dir = episode_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    json_path = metadata_dir / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    txt_path = metadata_dir / "metadata.txt"
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


def copy_episode_to_nas(episode_dir: Path) -> Path | None:
    """
    Copy the episode folder to the NAS destination.

    Args:
        episode_dir: Path to the episode folder to copy

    Returns:
        Path to the copied folder on NAS, or None if copy failed
    """
    if not NAS_DESTINATION.exists():
        print(f"Warning: NAS destination not accessible: {NAS_DESTINATION}")
        return None

    try:
        destination = NAS_DESTINATION / episode_dir.name
        print(f"Copying episode to NAS: {destination}")
        shutil.copytree(episode_dir, destination, dirs_exist_ok=True)
        print(f"  Episode copied to NAS successfully")
        return destination
    except Exception as e:
        print(f"Warning: Failed to copy episode to NAS: {e}")
        return None


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
    print(f"Using Chatterbox (fal.ai) with Voice Cloning")
    print(f"Hosts: {HOST_NAME} & {CO_HOST_NAME}")
    print(f"Output folder: {episode_dir}")
    print(f"{'='*60}\n")

    # Verify API keys and voice samples upfront
    gemini_client = get_gemini_client()
    get_fal_client()

    for speaker, sample_path in VOICE_SAMPLES.items():
        if not sample_path.exists():
            raise FileNotFoundError(f"Voice sample not found for {speaker}: {sample_path}")
        print(f"Voice sample for {speaker}: {sample_path.name}")

    # OPTIMIZATION: Run voice upload, script generation, and prompt processing in parallel
    # Voice upload is slow (network), script gen is slow (LLM), prompt processing is I/O bound
    print("\nStep 1: Uploading voice samples + generating script + processing prompt audio (parallel)...")

    voice_urls = None
    script = None
    processed_prompt_path = episode_dir / "prompt_processed.wav"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        voice_future = executor.submit(upload_voice_samples)
        script_future = executor.submit(transcribe_and_generate_script, gemini_client, prompt_audio_path)
        prompt_future = executor.submit(process_prompt_audio, prompt_audio_path, processed_prompt_path)

        # Wait for all to complete
        voice_urls = voice_future.result()
        print("  Voice samples uploaded")
        script = script_future.result()
        print("  Script generated")
        processed_prompt_path = prompt_future.result()
        print("  Prompt audio processed")

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
        user_prompt_audio=processed_prompt_path,
        intro_jingle=intro_jingle if intro_jingle.exists() else None,
        disclaimer_audio=DISCLAIMER_PATH if DISCLAIMER_PATH.exists() else None,
        outro_jingle=outro_jingle if outro_jingle.exists() else None,
    )

    # Cleanup intermediate files
    if dialogue_audio_path.exists():
        dialogue_audio_path.unlink()
    if processed_prompt_path.exists():
        processed_prompt_path.unlink()

    full_metadata = {
        'title': metadata['title'],
        'description': metadata['description'],
        'image_prompt': metadata.get('image_prompt', ''),
        'cover_art': [str(p) for p in cover_art_paths] if cover_art_paths else None,
        'episode_name': episode_name,
        'audio_file': str(episode_path),
        'script_file': str(script_path),
        'segments_count': len(segments),
        'tts_engine': 'chatterbox-fal',
        'voice_samples': {
            HOST_NAME: str(VOICE_SAMPLES[HOST_NAME]),
            CO_HOST_NAME: str(VOICE_SAMPLES[CO_HOST_NAME]),
        },
        'generated_at': datetime.now().isoformat(),
    }

    save_metadata_files(full_metadata, episode_dir)

    # Copy episode to NAS
    nas_path = copy_episode_to_nas(episode_dir)

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
    print(f"  - metadata/ (metadata.json, metadata.txt)")
    if cover_art_paths:
        print(f"  - images/")
        for cap in cover_art_paths:
            print(f"      - {cap.name}")
    print(f"  ({len(segments)} dialogue turns)")
    if nas_path:
        print(f"\nNAS COPY: {nas_path}")
    print(f"{'='*60}\n")

    return episode_path


def process_queue():
    """Process all audio files in the to-process queue."""
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    to_process = sorted([
        f for f in PROMPTS_TO_PROCESS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ])

    if not to_process:
        print("No audio files found in the to-process queue.")
        print(f"Add audio files to: {PROMPTS_TO_PROCESS_DIR}")
        return

    total_episodes = len(to_process)

    print(f"\n{'='*60}")
    print(f"  QUEUE STATUS: {total_episodes} episode(s) to generate")
    print(f"{'='*60}")
    for i, f in enumerate(to_process, 1):
        print(f"  [{i}] {f.name}")
    print(f"{'='*60}\n")

    successful = 0
    failed = 0

    for idx, prompt_path in enumerate(to_process, 1):
        print(f"\n{'#'*60}")
        print(f"  QUEUE: Processing episode {idx} of {total_episodes}")
        print(f"  FILE:  {prompt_path.name}")
        remaining = total_episodes - idx
        if remaining > 0:
            print(f"  REMAINING: {remaining} episode(s) after this")
        print(f"{'#'*60}")

        try:
            episode_name = prompt_path.stem
            episode_path = generate_podcast_episode(prompt_path, episode_name)

            # Delete the prompt file after successful processing
            prompt_path.unlink()
            print(f"Deleted processed prompt: {prompt_path.name}")
            successful += 1

        except Exception as e:
            print(f"Error processing {prompt_path.name}: {e}")
            print(f"  (prompt file kept for retry)")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    print(f"\n{'='*60}")
    print(f"  QUEUE COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful: {successful}")
    if failed > 0:
        print(f"  Failed:     {failed}")
    print(f"  Total:      {total_episodes}")
    print(f"{'='*60}\n")


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

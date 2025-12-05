# CLAUDE.md - My Weird Prompts

> **Note**: This validated pipeline has been integrated into the main My Weird Prompts website deployment repository at https://github.com/danielrosehill/My-Weird-Prompts

## Project Goal

Build a **semi-automated AI podcast workflow** for *My Weird Prompts* (A Human-AI Podcast Collaboration):
1. Takes human-recorded audio prompts from Daniel Rosehill (producer)
2. Sends them to Gemini for transcription and dialogue script generation
3. Generates voice-cloned TTS via Replicate (Chatterbox)
4. Assembles complete podcast episodes with intro, disclaimer, prompt, dialogue, and outro
5. Outputs normalized, publish-ready audio files

**Key differentiator**: The human's actual voice recording is included in the final episode, not just transcribed. This creates a hybrid human+AI podcast format.

---

## Quick Start

```bash
# Drop audio prompts into the queue
cp your-prompt.mp3 pipeline/prompts/to-process/

# Generate episodes
./generate.sh

# Output appears in pipeline/output/episodes/<episode-name>/
```

---

## Workflow Architecture

```
[Audio Prompt] -> [Gemini: Transcribe + Generate Script] -> [Chatterbox TTS via Replicate]
                                                                       |
                                                                       v
[Intro] + [Disclaimer] + [Original Prompt Audio] + [AI Dialogue] + [Outro]
                                                                       |
                                                                       v
                                            [Normalized MP3 + Metadata + Cover Art]
```

---

## Episode Format

All generated episodes follow this structure:

```
[Intro Jingle] -> [Disclaimer] -> [Daniel's Prompt] -> [AI Dialogue (~15 min)] -> [Outro Jingle]
```

**Hosts**: Corn & Herman (voice-cloned from samples in `config/voices/`)
**Producer**: Daniel Rosehill (hosts reference him by name, not as "a listener")

**Cost**: ~$0.40 per 15-minute episode (Replicate TTS + Gemini + cover art)

---

## File Structure

```
/pipeline/
  /generators/
    generate_episode.py     # Main generator (Chatterbox via Replicate)
    generate_disclaimer.py  # One-time disclaimer generator
    /archived/              # Legacy generators (Gemini, Kokoro, Resemble, etc.)
  /prompts/
    /to-process/            # Drop audio prompts here
    /done/                  # Processed prompts moved here
  /output/
    /episodes/              # Final rendered episodes (each in own folder)
  /show-elements/
    /mixed/                 # Pre-mixed show elements
      mixed-intro.mp3
      disclaimer.mp3        # AI-generated disclaimer
      mixed-outro.mp3

/config/
  /voices/                  # Voice samples for cloning (corn, herman)

/docs/
  /planning/                # Idea notes, show concept
  /reference/               # API docs, TTS research

generate.sh                 # Launcher script
record_prompt.py            # GUI for recording prompts
```

---

## Technical Details

### Audio Specifications
- Output format: MP3, 192kbps
- Loudness: -16 LUFS (podcast standard)
- Sample rate: 44.1kHz
- Normalization: EBU R128 two-pass

### Dependencies
- Python 3.11+ with venv
- ffmpeg (system-wide)
- API keys: `GEMINI_API_KEY`, `REPLICATE_API_TOKEN`

### Configuration
Key settings in `generate_episode.py`:
- `PODCAST_NAME = "My Weird Prompts"`
- `PRODUCER_NAME = "Daniel Rosehill"`
- `MAX_TTS_WORKERS = 4` - Parallel Replicate API calls
- `TARGET_LUFS = -16` - Podcast loudness standard
- Voice samples in `config/voices/{corn,herman}/wav/`

---

## Archived Generators

Previous TTS implementations are in `pipeline/generators/archived/`:
- `gemini_dialogue.py` - Gemini native TTS
- `kokoro_dialogue.py` - Local Kokoro TTS (ROCm)
- `resemble_dialogue.py` - Resemble AI direct API
- `openai_single_host.py` - Single host format
- `chatterbox_local_dialogue.py` - Local Chatterbox server

These can be restored if needed for cost savings (local) or different voice options.

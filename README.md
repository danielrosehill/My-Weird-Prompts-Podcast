# My Weird Prompts - Podcast Production Pipeline

![alt text](graphics/repo/2.png)

**[Listen on Spotify](https://open.spotify.com/show/4RlBls1ZQxs4ciREOR8vpU)**

The production pipeline for *My Weird Prompts*, a semi-automated AI podcast that combines human-recorded prompts with AI-generated responses, rendered as listenable audio episodes.

## About the Show

*My Weird Prompts* is a podcast where the host shares interesting, unusual, or thought-provoking prompts and AI responds with detailed, conversational explanations. The result is a hybrid human+AI podcast format that preserves the authenticity of the original question while leveraging AI for comprehensive responses.

## The Approach

This workflow addresses several limitations with existing AI podcast solutions:

### Why Not Notebook LM?
Notebook LM produces great content, but the podcast style tends toward a specific Americanized, California-esque host format that doesn't suit everyone's preferences.

### Why Not Pure TTS Workflows?
Previous N8N workflows (speech-to-text → LLM → text-to-speech) work but have issues:
- Quality TTS (like ElevenLabs) is expensive for regular 30-minute episodes
- Fully synthetic output lacks the human element

### The Solution
A hybrid approach:
1. **Human prompts** - Recorded audio prompts from the creator
2. **AI responses** - Multimodal AI generates podcast-style dialogue
3. **Combined output** - Final episode includes: intro jingle → disclaimer → human prompt → AI dialogue → outro jingle

## Workflow

```
[Audio Prompt Recording]
        ↓
[Gemini: Transcribe + Generate Script]
        ↓
[Chatterbox TTS via Replicate]
        ↓
[Assemble Episode]
   - Intro jingle
   - Disclaimer
   - Human prompt audio
   - AI dialogue (~15 min)
   - Outro jingle
        ↓
[Render Normalized MP3 + Metadata]
        ↓
[Upload to Spotify]
```

## Quick Start

```bash
# Drop audio prompts into the queue
cp your-prompt.mp3 pipeline/prompts/to-process/

# Generate episodes
./generate.sh

# Output appears in pipeline/output/episodes/<episode-name>/
```

## Cost

Approximately $0.40 per 15-minute episode (Replicate TTS + Gemini + cover art generation).

## Repository Structure

- `pipeline/generators/` - Episode generation scripts
- `pipeline/prompts/` - Audio prompt queue (to-process/done)
- `pipeline/output/episodes/` - Rendered episodes
- `pipeline/show-elements/` - Intro/outro jingles
- `config/voices/` - Voice samples for TTS cloning

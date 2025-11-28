# AI Podcast Experiment - Idea Notes

*Transcribed from theidea.mp3 on 28 Nov 2025*

---

## Overview

This repository is the latest iteration of an AI podcast experiment. The goal is to create a workflow where I (Daniel) record voice prompts, and those prompts get transformed into full podcast episodes with AI-generated responses.

## Background

I've had this working before in a reliable pipeline using N8N workflows, but I want to improve it. Existing solutions like NotebookLM are amazing, but the "Americanized California-esque host style" isn't the production style I want for my podcast.

## The Problem I'm Solving

- I come up with detailed prompts during the day and get great AI outputs
- I don't always have time to read/consume them in the moment
- As a new parent, audio is the most versatile format - I can listen at the gym, walking, or while minding our newborn
- I might want to do this a few times a day, so it needs to be cost-effective
- TTS quality matters a lot - 11Labs sounds great but gets expensive for 30-minute episodes

## Current Iteration's Key Insight

**Use my actual voice recordings in the podcast.** Instead of just transcribing my prompt and sending it to AI, why not include my original audio recording? The episode would:

1. Start with my voice (the actual prompt recording)
2. Cut to the TTS segment (AI response)

This makes it more human - it really is me coming up with prompts, then handing off to the AI.

## Proposed Workflow

1. Record prompt (audio, like this file)
2. Send to multimodal AI (Gemini) for transcription + response generation
3. Ideally one-shot: send audio file to Gemini saying "this is a prompt, generate a podcast episode responding to it"
4. Final assembly:
   - Intro jingle
   - My prompt (original audio)
   - Pause/transition
   - AI response (TTS)
   - Outro jingle
5. Render to normalized audio file

## Stretch Goal: Full End-to-End

The workflow could publish directly as a podcast episode with:
- Title
- Cover art
- Episode description

Transistor FM has an API for podcast publication, but manual upload to Spotify is acceptable for 1-2 episodes per day/week.

## Why Open Source This?

I come up with specific prompts about niche subjects and get good responses - maybe someone else would want to listen to them. I'm supportive of AI-generated podcasts as long as the information is good and the voice is pleasant to listen to.

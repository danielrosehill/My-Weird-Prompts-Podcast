# Stage 2 Notes - AI Podcast Experiment

**Recorded**: December 2025
**Source**: notes-2.mp3

---

## Vision & Motivation

I'm recording more context for the AI podcast experiment. I see so much potential in this idea—it's something I'm creating for myself, like a lot of my projects. During the day, I come up with very detailed questions about specific things. For example, today while out walking, I recorded one about control nets in generative AI.

The key insight: reading long AI responses at a specific time isn't enjoyable, but when they're packaged as an audio episode, you're immediately engaged because it's something you genuinely wanted to know. It pulls you in. It's a great way to consume long-form content that actually works.

**Core concept**: A personal way of consuming information and learning from AI—getting lengthy responses based on lengthy inputs, delivered as listenable content.

---

## Current State & What Works

### Replicate Chatterbox
- Very impressive model
- Cost calculations showed it wasn't too expensive per episode
- Voice cloning worked fine with zero-shot (best you can get with fake voices for imaginary characters)
- The voices were great, scripts were good and informative
- I've enjoyed listening to the generated episodes—some really good ones over the last few days

### The Two-Host Format
I've been using two characters: **Corn the Sloth** and **Herman the Donkey** (stuffed animals I've brought to life in this podcast). The two-participant banter is more fun than a single host—I get a kick out of hearing them discuss topics in their ridiculous AI way. This was partly inspired by NotebookLM's two-host approach.

### Audio Concatenation
Essential element: my original voice prompt gets concatenated into the final episode. This makes it less purely robotic—there's actually a human (me) sending in the prompt, which is what's really happening. It adds authenticity.

---

## What Didn't Work

### Local TTS (Kokoro, etc.)
- Very hard to get working well, especially for two-participant models
- AMD GPU caused one problem after another
- Abandoned this approach despite cost benefits

### Gemini's Built-in TTS
- Gemini is multimodal and can do text-to-speech, but doesn't support voice cloning
- I dislike all their voice models—they have this stuck, over-optimistic West California accent across genders that's very grating
- Those voices ruin the podcast for me

---

## The Efficiency Problem

Looking at Chatterbox runs, I noticed it was doing zero-shot voice cloning on **every single turn**—re-uploading voice samples each time. This seems very inefficient. It would make more sense to create something like a LoRA (as in image AI)—do it once for a proper voice clone and that's your voice.

---

## Pipeline Goals

### Current Workflow (Manual)
1. Record prompt on phone/computer
2. Send to pipeline (was using N8N + Voice Notes app webhook)
3. Get episode generated
4. Manually upload to Spotify for distribution

### Problems with Current Workflow
- Need to be at computer while scripts run
- Need to be at computer to upload via GitHub
- N8N is good but Python with AI agents opens more possibilities

### Ideal End-to-End Flow
1. **Input**: Record a 4-5 minute voice prompt while hands-free (at market, walking, etc.)
2. **Processing**: Pipeline handles transcription, AI response generation, TTS, audio concatenation—all in background
3. **Output**: Get notification (email?) saying "Your ControlNet podcast is ready, here's the link"
4. **Result**: Listen at the gym that night

**Key requirements**:
- Minimal intervention on my part
- Doesn't need to be immediate—an hour delay is fine
- I create a few and savor listening to them as they queue up
- Outputs needed: podcast episode MP3, title, description

---

## Publishing Challenge

### The Automation Gap
The only stage I don't want to automate is final publishing (for now). I publish to Spotify so I can:
- Listen to episodes myself
- Send links to friends ("Hey, I got this response from AI, here's a link")

### Transistor FM
- Only platform I found with programmatic publishing API
- ~$20/month subscription (drowning in SaaS already)
- Not sure they fully support what I'm envisioning

### Acceptable Alternatives
- Bundle episodes into Google Drive folder
- Object storage on server
- As long as I get a notification and have: episode file, title, description
- Then I can quickly pop those into any podcast publisher

---

## Deployment Questions

**What I'm asking**: How can we deploy this pipeline so I can:
- Send in a few prompts per day from phone or computer
- Get a few outputs per day
- Cost target: ~$3-4/day is worthwhile investment in learning content

### Environment Considerations
- Need persistent accessibility for webhooks
- Could be serverless (no experience with this)
- Could be pure code automation server
- Maybe hybrid model
- Building a custom app just for prompt capture seems overkill

### Why This Isn't Spam
On the outside, this might look like low-effort AI podcast spam. It's the opposite—I'm creating informative learning content for myself and anyone who finds it interesting, with clear AI-generated disclaimer. Claude's accuracy is ~99%, at least on par with educated human hosts. Humans get stuff wrong too.

---

## Summary of Next Steps

1. **Discard failed local approaches** - Focus on what works (Replicate Chatterbox)
2. **Find deployment solution** - Something that can receive webhooks and run the pipeline
3. **Optimize voice cloning** - Investigate if we can pre-create voice models instead of zero-shot every turn
4. **Publishing later** - Manual upload is fine for now, Transistor FM or similar could come later

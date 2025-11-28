#!/usr/bin/env python3
"""
Chatterbox TTS wrapper for podcast generation.
Runs inside the chatterbox-rocm Docker container.

Usage:
    python chatterbox_tts.py --text "Hello world" --output /app/output/speech.wav
    python chatterbox_tts.py --text "Hello world" --output /app/output/speech.wav --reference /app/references/voice.wav
"""

import argparse
import os
import sys
from pathlib import Path

# Set up environment for ROCm
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torchaudio as ta


def generate_tts(
    text: str,
    output_path: str,
    reference_audio: str = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
):
    """
    Generate TTS audio from text using Chatterbox.

    Args:
        text: The text to convert to speech
        output_path: Where to save the output audio
        reference_audio: Optional reference audio for voice cloning
        exaggeration: Emotion exaggeration level (0.0-1.0)
        cfg_weight: Classifier-free guidance weight
    """
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading Chatterbox model...")
    model = ChatterboxTTS.from_pretrained(device=device)

    print(f"Generating speech for: {text[:100]}{'...' if len(text) > 100 else ''}")

    # Generate audio
    if reference_audio and Path(reference_audio).exists():
        print(f"Using reference voice from: {reference_audio}")
        wav = model.generate(
            text,
            audio_prompt_path=reference_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    else:
        wav = model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ta.save(str(output_path), wav, model.sr)
    print(f"Audio saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS Generator")
    parser.add_argument("--text", "-t", required=True, help="Text to convert to speech")
    parser.add_argument("--output", "-o", required=True, help="Output audio file path")
    parser.add_argument("--reference", "-r", help="Reference audio for voice cloning")
    parser.add_argument("--exaggeration", "-e", type=float, default=0.5,
                        help="Emotion exaggeration (0.0-1.0, default 0.5)")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="CFG weight (default 0.5)")

    args = parser.parse_args()

    generate_tts(
        text=args.text,
        output_path=args.output,
        reference_audio=args.reference,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )


if __name__ == "__main__":
    main()

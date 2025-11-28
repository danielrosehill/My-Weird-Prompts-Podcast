#!/usr/bin/env python3
"""
Fish Speech TTS wrapper for podcast generation.
Runs inside the fish-speech-rocm Docker container.

Usage:
    python fish_tts.py --text "Hello world" --output /app/output/speech.wav
    python fish_tts.py --text "Hello world" --output /app/output/speech.wav --reference /app/references/voice.wav
"""

import argparse
import os
import sys
from pathlib import Path

# Set up environment for ROCm
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torchaudio

def download_model_if_needed():
    """Download the Fish Speech model if not present."""
    model_path = Path("/app/checkpoints/openaudio-s1-mini")
    if not model_path.exists() or not (model_path / "codec.pth").exists():
        print("Downloading Fish Speech model...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="fishaudio/openaudio-s1-mini",
            local_dir=str(model_path),
            local_dir_use_symlinks=False
        )
        print("Model downloaded successfully!")
    return model_path


def generate_tts(
    text: str,
    output_path: str,
    reference_audio: str = None,
    reference_text: str = None,
):
    """
    Generate TTS audio from text using Fish Speech.

    Args:
        text: The text to convert to speech
        output_path: Where to save the output audio
        reference_audio: Optional reference audio for voice cloning
        reference_text: Optional transcript of reference audio
    """
    from fish_speech.inference_engine import TTSInferenceEngine

    model_path = download_model_if_needed()

    print(f"Initializing TTS engine...")
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Initialize the inference engine
    engine = TTSInferenceEngine(
        llama_checkpoint_path=str(model_path / "model.pth"),
        decoder_checkpoint_path=str(model_path / "codec.pth"),
        decoder_config_name="openaudio_s1",
        llama_config_name="dual_ar_8_codebook_mid",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        compile=False,  # Disable compile for ROCm compatibility
    )

    print(f"Generating speech for: {text[:100]}...")

    # Generate audio
    if reference_audio and Path(reference_audio).exists():
        print(f"Using reference voice from: {reference_audio}")
        # Load reference audio
        ref_audio, sr = torchaudio.load(reference_audio)
        if sr != 24000:
            ref_audio = torchaudio.functional.resample(ref_audio, sr, 24000)

        audio = engine.inference(
            text=text,
            reference_audio=ref_audio,
            reference_text=reference_text or "",
        )
    else:
        # Generate with default voice
        audio = engine.inference(text=text)

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # audio is a tensor, save it
    torchaudio.save(str(output_path), audio.cpu(), 24000)
    print(f"Audio saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fish Speech TTS Generator")
    parser.add_argument("--text", "-t", required=True, help="Text to convert to speech")
    parser.add_argument("--output", "-o", required=True, help="Output audio file path")
    parser.add_argument("--reference", "-r", help="Reference audio for voice cloning")
    parser.add_argument("--reference-text", help="Transcript of reference audio")
    parser.add_argument("--download-only", action="store_true", help="Only download the model")

    args = parser.parse_args()

    if args.download_only:
        download_model_if_needed()
        return

    generate_tts(
        text=args.text,
        output_path=args.output,
        reference_audio=args.reference,
        reference_text=args.reference_text,
    )


if __name__ == "__main__":
    main()

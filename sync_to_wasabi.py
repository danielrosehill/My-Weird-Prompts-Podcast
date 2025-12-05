#!/usr/bin/env python3
"""
Sync migrated podcast files to Wasabi bucket.

Uploads:
- Audio files from migration_cache/audio/
- Cover art from migration_cache/cover_art.jpg
- Markdown posts from migrated_posts/
- Podcast metadata from podcast_data.json

Usage:
    python sync_to_wasabi.py
"""

import os
import json
import boto3
from pathlib import Path
from botocore.config import Config

# Configuration
BUCKET_NAME = "myweirdprompts"
WASABI_ENDPOINT = "https://s3.eu-central-2.wasabisys.com"  # EU Central region

# Paths
AUDIO_CACHE_DIR = Path("./migration_cache/audio")
COVER_ART_PATH = Path("./migration_cache/cover_art.jpg")
POSTS_DIR = Path("./migrated_posts")
PODCAST_DATA_FILE = Path("./podcast_data.json")

# S3 folder structure in Wasabi
FOLDERS = {
    "episodes": "episodes/",
    "posts": "posts/",
    "cover_art": "assets/",
    "metadata": "",
}


def get_wasabi_client():
    """Create Wasabi S3 client using AWS credentials."""
    # Wasabi uses AWS-compatible credentials from ~/.aws/credentials
    return boto3.client(
        's3',
        endpoint_url=WASABI_ENDPOINT,
        config=Config(signature_version='s3v4')
    )


def get_content_type(filename: str) -> str:
    """Get content type for file."""
    ext = Path(filename).suffix.lower()
    content_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.md': 'text/markdown',
        '.json': 'application/json',
    }
    return content_types.get(ext, 'application/octet-stream')


def upload_file(client, filepath: Path, key: str) -> bool:
    """Upload a single file to Wasabi."""
    try:
        content_type = get_content_type(filepath.name)

        client.upload_file(
            str(filepath),
            BUCKET_NAME,
            key,
            ExtraArgs={'ContentType': content_type}
        )
        print(f"  [uploaded] {key}")
        return True
    except Exception as e:
        print(f"  [error] {key}: {e}")
        return False


def sync_audio_files(client):
    """Sync all audio files to Wasabi."""
    print("\n=== Syncing Audio Files ===")

    if not AUDIO_CACHE_DIR.exists():
        print("No audio cache directory found")
        return 0

    audio_files = list(AUDIO_CACHE_DIR.glob("*.mp3")) + \
                  list(AUDIO_CACHE_DIR.glob("*.wav")) + \
                  list(AUDIO_CACHE_DIR.glob("*.m4a"))

    print(f"Found {len(audio_files)} audio files")

    uploaded = 0
    for filepath in audio_files:
        key = f"{FOLDERS['episodes']}{filepath.name}"
        if upload_file(client, filepath, key):
            uploaded += 1

    return uploaded


def sync_cover_art(client):
    """Sync cover art to Wasabi."""
    print("\n=== Syncing Cover Art ===")

    if not COVER_ART_PATH.exists():
        print("No cover art found")
        return 0

    key = f"{FOLDERS['cover_art']}cover-art.jpg"
    return 1 if upload_file(client, COVER_ART_PATH, key) else 0


def sync_markdown_posts(client):
    """Sync markdown blog posts to Wasabi."""
    print("\n=== Syncing Markdown Posts ===")

    if not POSTS_DIR.exists():
        print("No posts directory found")
        return 0

    md_files = list(POSTS_DIR.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    uploaded = 0
    for filepath in md_files:
        key = f"{FOLDERS['posts']}{filepath.name}"
        if upload_file(client, filepath, key):
            uploaded += 1

    return uploaded


def sync_metadata(client):
    """Sync podcast metadata JSON to Wasabi."""
    print("\n=== Syncing Metadata ===")

    if not PODCAST_DATA_FILE.exists():
        print("No podcast data file found")
        return 0

    key = "podcast_data.json"
    return 1 if upload_file(client, PODCAST_DATA_FILE, key) else 0


def main():
    print("=" * 60)
    print("Wasabi Sync: My Weird Prompts")
    print("=" * 60)
    print(f"Bucket: {BUCKET_NAME}")
    print(f"Endpoint: {WASABI_ENDPOINT}")

    client = get_wasabi_client()

    # Check bucket exists
    try:
        client.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' accessible")
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return

    # Sync all content
    results = {
        'audio': sync_audio_files(client),
        'cover_art': sync_cover_art(client),
        'posts': sync_markdown_posts(client),
        'metadata': sync_metadata(client),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"Audio files: {results['audio']}")
    print(f"Cover art: {results['cover_art']}")
    print(f"Markdown posts: {results['posts']}")
    print(f"Metadata: {results['metadata']}")
    print(f"Total files: {sum(results.values())}")

    # Generate Wasabi URLs
    print("\n=== Wasabi URLs ===")
    print(f"Episodes: https://s3.wasabisys.com/{BUCKET_NAME}/episodes/")
    print(f"Cover Art: https://s3.wasabisys.com/{BUCKET_NAME}/assets/cover-art.jpg")


if __name__ == "__main__":
    main()

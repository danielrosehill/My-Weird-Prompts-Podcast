#!/usr/bin/env python3
"""
Migrate podcast from Anchor/Spotify to self-hosted blog.

Downloads all episodes from Anchor RSS, uploads audio to Cloudinary,
and generates blog post markdown files for the My-Weird-Prompts Astro site.

Usage:
    python migrate_from_anchor.py

Output:
    - Audio files uploaded to Cloudinary
    - Markdown files in ./migrated_posts/
    - podcast_data.json with all episode metadata
"""

import os
import re
import json
import hashlib
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ANCHOR_RSS_URL = "https://anchor.fm/s/1082efa78/podcast/rss"
OUTPUT_DIR = Path("./migrated_posts")
AUDIO_CACHE_DIR = Path("./migration_cache/audio")
PODCAST_DATA_FILE = Path("./podcast_data.json")

# Cloudinary config from environment
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "my-weird-prompts/episodes")

# Parse Cloudinary URL if provided
if CLOUDINARY_URL:
    # Format: cloudinary://API_KEY:API_SECRET@CLOUD_NAME
    match = re.match(r'cloudinary://(\d+):([^@]+)@(.+)', CLOUDINARY_URL)
    if match:
        cloudinary.config(
            cloud_name=match.group(3),
            api_key=match.group(1),
            api_secret=match.group(2),
            secure=True
        )
        print(f"Cloudinary configured: {match.group(3)}")
    else:
        print("Warning: Could not parse CLOUDINARY_URL")


def fetch_rss_feed(url: str) -> str:
    """Fetch the RSS feed XML."""
    print(f"Fetching RSS feed from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_duration(duration_str: str) -> int:
    """Parse duration string to seconds."""
    if not duration_str:
        return 0

    parts = duration_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    else:
        try:
            return int(parts[0])
        except ValueError:
            return 0


def format_duration(seconds: int) -> str:
    """Format seconds to MM:SS or HH:MM:SS."""
    if seconds >= 3600:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"


def parse_rss_feed(xml_content: str) -> dict:
    """Parse RSS feed and extract show + episode data."""
    # Handle iTunes namespace
    namespaces = {
        'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
        'atom': 'http://www.w3.org/2005/Atom',
        'content': 'http://purl.org/rss/1.0/modules/content/'
    }

    root = ET.fromstring(xml_content)
    channel = root.find('channel')

    # Extract show-level metadata
    show_data = {
        'title': channel.findtext('title', ''),
        'description': channel.findtext('description', ''),
        'author': channel.findtext('itunes:author', '', namespaces) or channel.findtext('author', ''),
        'language': channel.findtext('language', 'en'),
        'link': channel.findtext('link', ''),
        'cover_art': None,
        'categories': [],
        'episodes': []
    }

    # Get cover art
    itunes_image = channel.find('itunes:image', namespaces)
    if itunes_image is not None:
        show_data['cover_art'] = itunes_image.get('href')
    else:
        image = channel.find('image')
        if image is not None:
            show_data['cover_art'] = image.findtext('url', '')

    # Get categories
    for cat in channel.findall('itunes:category', namespaces):
        cat_text = cat.get('text')
        if cat_text:
            show_data['categories'].append(cat_text)
            # Check for subcategories
            for subcat in cat.findall('itunes:category', namespaces):
                subcat_text = subcat.get('text')
                if subcat_text:
                    show_data['categories'].append(subcat_text)

    # Parse episodes
    for item in channel.findall('item'):
        episode = {
            'title': item.findtext('title', ''),
            'description': item.findtext('description', ''),
            'content': item.findtext('content:encoded', '', namespaces) or item.findtext('description', ''),
            'pub_date': item.findtext('pubDate', ''),
            'guid': item.findtext('guid', ''),
            'audio_url': None,
            'audio_type': None,
            'audio_length': 0,
            'duration': item.findtext('itunes:duration', '', namespaces),
            'episode_image': None,
            'episode_number': item.findtext('itunes:episode', '', namespaces),
            'season_number': item.findtext('itunes:season', '', namespaces),
            'keywords': item.findtext('itunes:keywords', '', namespaces),
        }

        # Get audio enclosure
        enclosure = item.find('enclosure')
        if enclosure is not None:
            episode['audio_url'] = enclosure.get('url')
            episode['audio_type'] = enclosure.get('type', 'audio/mpeg')
            try:
                episode['audio_length'] = int(enclosure.get('length', 0))
            except ValueError:
                episode['audio_length'] = 0

        # Get episode-specific image
        ep_image = item.find('itunes:image', namespaces)
        if ep_image is not None:
            episode['episode_image'] = ep_image.get('href')

        # Parse pub_date to ISO format
        if episode['pub_date']:
            try:
                # Parse RFC 2822 date format
                dt = datetime.strptime(episode['pub_date'], '%a, %d %b %Y %H:%M:%S %z')
                episode['pub_date_iso'] = dt.strftime('%Y-%m-%d')
                episode['pub_date_full'] = dt.isoformat()
            except ValueError:
                try:
                    dt = datetime.strptime(episode['pub_date'], '%a, %d %b %Y %H:%M:%S %Z')
                    episode['pub_date_iso'] = dt.strftime('%Y-%m-%d')
                    episode['pub_date_full'] = dt.isoformat()
                except ValueError:
                    episode['pub_date_iso'] = datetime.now().strftime('%Y-%m-%d')
                    episode['pub_date_full'] = episode['pub_date']

        # Calculate duration in seconds
        episode['duration_seconds'] = parse_duration(episode['duration'])
        episode['duration_formatted'] = format_duration(episode['duration_seconds'])

        show_data['episodes'].append(episode)

    print(f"Parsed {len(show_data['episodes'])} episodes from feed")
    return show_data


def download_audio(episode: dict, cache_dir: Path) -> Path | None:
    """Download audio file to cache directory."""
    if not episode.get('audio_url'):
        return None

    # Create filename from GUID or title
    slug = slugify(episode['title'])
    ext = '.mp3'  # Default to mp3
    if episode.get('audio_type') == 'audio/x-m4a':
        ext = '.m4a'
    elif episode.get('audio_type') == 'audio/ogg':
        ext = '.ogg'

    filename = f"{slug}{ext}"
    filepath = cache_dir / filename

    # Skip if already downloaded
    if filepath.exists():
        print(f"  [cached] {filename}")
        return filepath

    try:
        print(f"  Downloading: {episode['title'][:50]}...")
        response = requests.get(episode['audio_url'], stream=True, timeout=120)
        response.raise_for_status()

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  [downloaded] {filename} ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
        return filepath

    except Exception as e:
        print(f"  [error] Failed to download {episode['title']}: {e}")
        return None


def upload_to_cloudinary(filepath: Path, episode: dict) -> str | None:
    """Upload audio file to Cloudinary and return URL."""
    if not filepath or not filepath.exists():
        return None

    try:
        slug = slugify(episode['title'])
        public_id = f"{CLOUDINARY_FOLDER}/{slug}"

        print(f"  Uploading to Cloudinary: {filepath.name}...")

        result = cloudinary.uploader.upload(
            str(filepath),
            resource_type="video",  # Cloudinary uses "video" for audio
            public_id=public_id,
            overwrite=False,  # Don't re-upload if exists
            folder=None,  # Already included in public_id
        )

        url = result.get('secure_url') or result.get('url')
        print(f"  [uploaded] {url}")
        return url

    except Exception as e:
        # Check if it's a "already exists" error
        if "already exists" in str(e).lower():
            # Construct the URL
            cloud_name = cloudinary.config().cloud_name
            url = f"https://res.cloudinary.com/{cloud_name}/video/upload/{public_id}.mp3"
            print(f"  [exists] {url}")
            return url

        print(f"  [error] Failed to upload {filepath.name}: {e}")
        return None


def slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    text = text.strip('-')
    return text[:80]  # Limit length


def extract_tags_from_description(description: str) -> list[str]:
    """Extract relevant tags from episode description."""
    # Common AI/tech keywords to look for
    keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'llm', 'gpt',
        'claude', 'chatgpt', 'openai', 'anthropic', 'google', 'gemini',
        'automation', 'workflow', 'productivity', 'coding', 'programming',
        'python', 'javascript', 'api', 'prompt', 'prompting',
        'voice', 'audio', 'podcast', 'technology', 'tech',
        'data', 'analysis', 'security', 'privacy', 'gpu', 'hardware',
        'cloud', 'infrastructure', 'devops', 'docker', 'linux',
        'home assistant', 'smart home', 'iot',
    ]

    description_lower = description.lower()
    found_tags = []

    for keyword in keywords:
        if keyword in description_lower:
            # Capitalize properly
            tag = keyword.title().replace(' ', '')
            if tag not in found_tags:
                found_tags.append(tag)

    return found_tags[:6]  # Limit to 6 tags


def clean_html(html: str) -> str:
    """Remove HTML tags from string."""
    clean = re.sub(r'<[^>]+>', '', html)
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    clean = clean.replace('&quot;', '"')
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def generate_blog_post(episode: dict, cloudinary_url: str, show_data: dict) -> str:
    """Generate markdown blog post for an episode."""

    title = episode['title']
    description = clean_html(episode['description'])
    content = clean_html(episode.get('content', description))
    pub_date = episode.get('pub_date_iso', datetime.now().strftime('%Y-%m-%d'))
    duration = episode.get('duration_formatted', '0:00')
    tags = extract_tags_from_description(description)

    # Use episode image or show cover art
    hero_image = episode.get('episode_image') or show_data.get('cover_art', '')

    # Generate frontmatter
    frontmatter = f'''---
title: "{title.replace('"', '\\"')}"
description: "{description[:200].replace('"', '\\"')}..."
pubDate: "{pub_date}"
heroImage: "{hero_image}"
tags: {json.dumps(tags)}
prompt: "Episode from My Weird Prompts podcast"
podcastAudioUrl: "{cloudinary_url}"
podcastDuration: "{duration}"
aiGenerated: true
migratedFromAnchor: true
originalGuid: "{episode.get('guid', '')}"
---

## About This Episode

{description}

## Listen

<audio controls src="{cloudinary_url}" style="width: 100%;">
  Your browser does not support the audio element.
</audio>

**Duration:** {duration}

---

*This episode was migrated from the original podcast feed. It features AI-generated dialogue exploring prompts and questions submitted by Daniel Rosehill.*
'''

    return frontmatter


def main():
    """Main migration function."""
    print("=" * 60)
    print("Podcast Migration: Anchor -> Self-Hosted")
    print("=" * 60)

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch and parse RSS
    xml_content = fetch_rss_feed(ANCHOR_RSS_URL)
    show_data = parse_rss_feed(xml_content)

    print(f"\nShow: {show_data['title']}")
    print(f"Episodes: {len(show_data['episodes'])}")
    print(f"Cover Art: {show_data['cover_art']}")

    # Download cover art
    if show_data['cover_art']:
        cover_path = AUDIO_CACHE_DIR.parent / "cover_art.jpg"
        if not cover_path.exists():
            print("\nDownloading cover art...")
            try:
                response = requests.get(show_data['cover_art'], timeout=30)
                response.raise_for_status()
                cover_path.write_bytes(response.content)
                print(f"  Saved: {cover_path}")

                # Upload cover art to Cloudinary
                result = cloudinary.uploader.upload(
                    str(cover_path),
                    resource_type="image",
                    public_id=f"{CLOUDINARY_FOLDER}/cover-art",
                    overwrite=True
                )
                show_data['cover_art_cloudinary'] = result.get('secure_url')
                print(f"  Uploaded cover art: {show_data['cover_art_cloudinary']}")
            except Exception as e:
                print(f"  Error downloading cover art: {e}")

    # Process episodes
    print("\n" + "=" * 60)
    print("Processing Episodes")
    print("=" * 60)

    migrated_episodes = []

    for i, episode in enumerate(show_data['episodes'], 1):
        print(f"\n[{i}/{len(show_data['episodes'])}] {episode['title'][:60]}...")

        # Download audio
        audio_path = download_audio(episode, AUDIO_CACHE_DIR)

        # Upload to Cloudinary
        cloudinary_url = None
        if audio_path:
            cloudinary_url = upload_to_cloudinary(audio_path, episode)

        if cloudinary_url:
            episode['cloudinary_url'] = cloudinary_url

            # Generate blog post
            markdown = generate_blog_post(episode, cloudinary_url, show_data)

            # Save markdown file
            slug = slugify(episode['title'])
            filename = f"{episode.get('pub_date_iso', 'unknown')}-{slug}.md"
            filepath = OUTPUT_DIR / filename
            filepath.write_text(markdown, encoding='utf-8')

            episode['blog_post_file'] = str(filepath)
            migrated_episodes.append(episode)
            print(f"  [saved] {filename}")
        else:
            print(f"  [skipped] No audio URL available")

    # Save full podcast data
    show_data['migrated_episodes'] = migrated_episodes
    PODCAST_DATA_FILE.write_text(json.dumps(show_data, indent=2), encoding='utf-8')
    print(f"\nSaved podcast data to {PODCAST_DATA_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"Total episodes: {len(show_data['episodes'])}")
    print(f"Migrated: {len(migrated_episodes)}")
    print(f"Blog posts: {OUTPUT_DIR}")
    print(f"Podcast data: {PODCAST_DATA_FILE}")

    print("\nNext steps:")
    print("1. Review the generated markdown files in ./migrated_posts/")
    print("2. Copy them to ~/repos/github/My-Weird-Prompts/code/frontend/src/content/blog/")
    print("3. Create the podcast RSS feed endpoint")
    print("4. Deploy and test")


if __name__ == "__main__":
    main()

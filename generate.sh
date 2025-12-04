#!/bin/bash
#
# AI Podcast Generator - Launcher Script
# Allows user to select which generation format to use for processing prompts
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATORS_DIR="$SCRIPT_DIR/pipeline/generators"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Check for virtual environment
if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Please create it with: uv venv && uv pip install -r requirements.txt"
    exit 1
fi

# Display menu
echo "======================================"
echo "   AI Podcast Generator"
echo "======================================"
echo ""
echo "Select a generation format:"
echo ""
echo "  1) Gemini Dialogue (~15 min)"
echo "     Two AI hosts discuss the topic using Gemini's native TTS"
echo "     Cost: Low (~\$0.10/episode)"
echo ""
echo "  2) Chatterbox Dialogue (~15 min) [REPLICATE]"
echo "     Voice-cloned hosts (Corn & Herman) via Replicate"
echo "     Cost: Low (~\$1.88/episode)"
echo ""
echo "  3) Chatterbox LOCAL Dialogue (~15 min) [RECOMMENDED]"
echo "     Voice-cloned hosts via LOCAL Chatterbox server (ROCm)"
echo "     Cost: FREE (requires docker: chatterbox-tts)"
echo ""
echo "  4) Resemble Dialogue (~15 min)"
echo "     Two AI hosts (Corn & Herman) via Resemble AI direct API"
echo "     Cost: HIGH (~\$5-6/episode)"
echo ""
echo "  5) OpenAI Single Host (~2-4 min)"
echo "     Single AI host responds to your prompt"
echo "     Cost: Free (edge-tts) or Low (OpenAI TTS)"
echo ""
echo "  q) Quit"
echo ""

read -p "Enter choice [1-5, q]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Gemini Dialogue generator..."
        "$VENV_PYTHON" "$GENERATORS_DIR/gemini_dialogue.py" "$@"
        ;;
    2)
        echo ""
        echo "Starting Chatterbox Dialogue generator (Replicate)..."
        "$VENV_PYTHON" "$GENERATORS_DIR/chatterbox_dialogue.py" "$@"
        ;;
    3)
        echo ""
        echo "Starting Chatterbox LOCAL Dialogue generator..."
        # Check if chatterbox-tts container is running
        if ! docker ps --format '{{.Names}}' | grep -q 'chatterbox-tts'; then
            echo "Warning: chatterbox-tts container not running."
            echo "Starting it now..."
            docker start chatterbox-tts 2>/dev/null || {
                echo "Error: Could not start chatterbox-tts container."
                echo "Create it first with the appropriate docker run command."
                exit 1
            }
            echo "Waiting for server to initialize..."
            sleep 5
        fi
        "$VENV_PYTHON" "$GENERATORS_DIR/chatterbox_local_dialogue.py" "$@"
        ;;
    4)
        echo ""
        echo "Starting Resemble Dialogue generator..."
        "$VENV_PYTHON" "$GENERATORS_DIR/resemble_dialogue.py" "$@"
        ;;
    5)
        echo ""
        echo "Starting OpenAI Single Host generator..."
        "$VENV_PYTHON" "$GENERATORS_DIR/openai_single_host.py" "$@"
        ;;
    q|Q)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

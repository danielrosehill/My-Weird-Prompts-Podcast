#!/usr/bin/env python3
"""
Modern GUI for recording podcast prompts.
Uses PySide6 (Qt) for the interface and sounddevice for recording.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).parent
PROMPTS_DIR = PROJECT_ROOT / "prompts" / "to-process"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100


class RecorderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.audio_data = []
        self.output_file = None

        self.setup_ui()
        self.setup_timer()
        self.load_devices()

    def setup_ui(self):
        self.setWindowTitle("Podcast Prompt Recorder")
        self.setFixedSize(480, 480)

        # Apply dark theme (Catppuccin Mocha-inspired)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
            }
            QLineEdit {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 10px 12px;
                color: #cdd6f4;
                font-size: 13px;
            }
            QLineEdit::placeholder {
                color: #6c7086;
            }
            QLineEdit:focus {
                border: 2px solid #89b4fa;
            }
            QComboBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 8px;
                padding: 10px 12px;
                color: #cdd6f4;
                font-size: 13px;
            }
            QComboBox:focus {
                border: 2px solid #89b4fa;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 12px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #cdd6f4;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 10px;
                padding: 14px 28px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #585b70;
            }
            QPushButton#recordBtn {
                background-color: #a6e3a1;
                color: #1e1e2e;
            }
            QPushButton#recordBtn:hover {
                background-color: #94e2d5;
            }
            QPushButton#recordBtn:disabled {
                background-color: #313244;
                color: #585b70;
            }
            QPushButton#stopBtn {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton#stopBtn:hover {
                background-color: #eba0ac;
            }
            QPushButton#stopBtn:disabled {
                background-color: #313244;
                color: #585b70;
            }
            QPushButton#saveBtn {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QPushButton#saveBtn:hover {
                background-color: #b4befe;
            }
            QPushButton#saveBtn:disabled {
                background-color: #313244;
                color: #585b70;
            }
            QPushButton#discardBtn {
                background-color: #6c7086;
                color: #cdd6f4;
            }
            QPushButton#discardBtn:hover {
                background-color: #7f849c;
            }
            QPushButton#discardBtn:disabled {
                background-color: #313244;
                color: #585b70;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("🎙️ Record Podcast Prompt")
        title.setFont(QFont("", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Device selection
        device_label = QLabel("Microphone")
        device_label.setFont(QFont("", 11))
        layout.addWidget(device_label)

        self.device_combo = QComboBox()
        layout.addWidget(self.device_combo)

        # Episode name
        name_label = QLabel("Episode Name (optional)")
        name_label.setFont(QFont("", 11))
        layout.addWidget(name_label)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Leave empty for timestamp-based name")
        layout.addWidget(self.name_input)

        layout.addSpacing(20)

        # Timer display
        self.timer_label = QLabel("00:00")
        self.timer_label.setFont(QFont("", 52, QFont.Bold))
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("color: #f5e0dc;")
        self.timer_label.setFixedHeight(70)
        layout.addWidget(self.timer_label)

        # Status - positioned below timer with clear spacing
        self.status_label = QLabel("Ready to record")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("", 11))
        self.status_label.setStyleSheet("color: #a6adc8;")
        self.status_label.setFixedHeight(25)
        layout.addWidget(self.status_label)

        layout.addSpacing(15)

        # Buttons - Row 1: Record / Stop
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        self.record_btn = QPushButton("🎤 Record")
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setMinimumHeight(50)
        btn_layout.addWidget(self.record_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(50)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # Buttons - Row 2: Save / Discard (shown after recording)
        btn_layout2 = QHBoxLayout()
        btn_layout2.setSpacing(16)

        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self.save_recording)
        self.save_btn.setMinimumHeight(50)
        self.save_btn.setEnabled(False)
        btn_layout2.addWidget(self.save_btn)

        self.discard_btn = QPushButton("🗑 Discard")
        self.discard_btn.setObjectName("discardBtn")
        self.discard_btn.clicked.connect(self.discard_recording)
        self.discard_btn.setMinimumHeight(50)
        self.discard_btn.setEnabled(False)
        btn_layout2.addWidget(self.discard_btn)

        layout.addLayout(btn_layout2)

    def setup_timer(self):
        self.seconds = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

    def load_devices(self):
        """Load available input devices."""
        devices = sd.query_devices()
        input_devices = []
        default_idx = 0

        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                # Filter out virtual/monitor devices, keep real hardware
                name = d['name']
                if any(x in name.lower() for x in ['monitor', 'spdif', 'speex', 'upmix', 'vdownmix']):
                    continue
                input_devices.append((i, name))

        for combo_idx, (device_idx, name) in enumerate(input_devices):
            self.device_combo.addItem(name, device_idx)
            # Default to Samson Q2U if found
            if 'samson' in name.lower() or 'q2u' in name.lower():
                default_idx = combo_idx

        if input_devices:
            self.device_combo.setCurrentIndex(default_idx)

    def start_recording(self):
        """Start recording audio."""
        self.audio_data = []
        self.seconds = 0
        self.recording = True

        # Generate filename
        name = self.name_input.text().strip()
        if not name:
            name = datetime.now().strftime("prompt_%Y%m%d_%H%M%S")
        name = "".join(c for c in name if c.isalnum() or c in "._- ").replace(" ", "_")
        self.output_file = PROMPTS_DIR / f"{name}.wav"

        device_idx = self.device_combo.currentData()

        # Start recording stream
        def audio_callback(indata, frames, time, status):
            if self.recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=device_idx,
            channels=1,
            callback=audio_callback
        )
        self.stream.start()

        self.timer.start(1000)
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🔴 Recording...")
        self.status_label.setStyleSheet("color: #f38ba8;")

    def stop_recording(self):
        """Stop recording - wait for user to save or discard."""
        self.recording = False
        self.timer.stop()

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if self.audio_data:
            mins, secs = divmod(self.seconds, 60)
            self.status_label.setText(f"Recorded {mins:02d}:{secs:02d} - Save or Discard?")
            self.status_label.setStyleSheet("color: #f9e2af;")
            self.save_btn.setEnabled(True)
            self.discard_btn.setEnabled(True)

        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    def save_recording(self):
        """Save the recorded audio."""
        if self.audio_data:
            audio = np.concatenate(self.audio_data)
            sf.write(str(self.output_file), audio, SAMPLE_RATE)
            self.status_label.setText(f"✓ Saved: {self.output_file.name}")
            self.status_label.setStyleSheet("color: #a6e3a1;")

        self._reset_for_new_recording()

    def discard_recording(self):
        """Discard the recorded audio."""
        self.audio_data = []
        self.status_label.setText("Recording discarded")
        self.status_label.setStyleSheet("color: #a6adc8;")
        self._reset_for_new_recording()

    def _reset_for_new_recording(self):
        """Reset UI for a new recording."""
        self.audio_data = []
        self.seconds = 0
        self.timer_label.setText("00:00")
        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)

    def update_timer(self):
        """Update the timer display."""
        self.seconds += 1
        mins, secs = divmod(self.seconds, 60)
        self.timer_label.setText(f"{mins:02d}:{secs:02d}")



def main():
    app = QApplication(sys.argv)
    window = RecorderWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

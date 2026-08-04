
# Audio2Midi
Python script to convert MP3 to MIDI

# Usage

python main.py input.mp3 -o output.mid

# Requirements

- ffmpeg
- pydub
- numpy
- scipy
- mido
- numba
- audioop-lts

# Installation

I use uv, but any package manager works:

git clone https://github.com/cylin577/Audio2Midi
cd Audio2Midi
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
    

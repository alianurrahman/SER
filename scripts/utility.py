from __future__ import unicode_literals

import spleeter
import yt_dlp as youtube_dl
from pydub import AudioSegment
from pydub.silence import split_on_silence
import csv
from pathlib import Path
from termcolor import colored
import os
import yaml

def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def convert_audio(audio_file):
    """
    Corrects the channels, sample rate, and sample width of the audios.
    Replaces the original audio file with the one generated.
    """
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2) # 2 corresponds to 16-bit sample width in Pydub
    sound.export(audio_file, format ="wav")

ydl_opts = {
    "format": "bestaudio/best",
    "audio-format": "wav",
    "outtmpl": "audio.wav",
    "noplaylist" : True
} # customizing the downloaded audio for our needs
link_num = 1 # iterates over the links in the TXT file
links_file = "/content/links.txt" # File containing links to YouTube videos
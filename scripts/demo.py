# -*- coding: utf-8 -*-

"""
This script is served as model demonstration build upon gradio

Usage:
    python3 ./scripts/demo.py

"""

import gradio as gr
from transformers import pipeline
import time

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio, state=""):
    time.sleep(3)
    text = transcriber(audio)["text"]
    state += text + " "
    return state, state

# GUI Component
gui_params = {
    "fn": transcribe,
    "inputs": [gr.Audio(sources=["microphone",], type="filepath", streaming=True), "state"],
    "outputs": ["textbox", "state"],
    "live":True,
    "allow_flagging": "never",
    "examples": "./../data/examples"

}
demo = gr.Interface(**gui_params)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
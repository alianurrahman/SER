# -*- coding: utf-8 -*-

"""
This script is used to do prediction based on trained model

Usage:
    python3 ./scripts/predict.py

"""
from huggingface_hub import notebook_login


if __name__ == '__main__':
    notebook_login()
    trainer.push_to_hub()

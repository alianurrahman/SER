# -*- coding: utf-8 -*-

"""
This script is used to do deploy based on latest trained model

Usage:
    python3 ./scripts/deploy.py

"""
from huggingface_hub import notebook_login

from train import train

def deploy():
    notebook_login()
    model = train()
    model.push_to_hub()

if __name__ == '__main__':
    deploy()

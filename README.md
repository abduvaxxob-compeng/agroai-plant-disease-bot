# AgroAI — End-to-End Plant Disease Diagnostic System

An AI-powered Telegram bot designed for rapid, accessible crop disease diagnostics using leaf imagery. Built for farmers and smallholder growers in Uzbekistan to enable early disease detection in Uzbek and Russian without requiring standalone mobile app installations.

## System Architecture

- Input: Farmer Smartphone (Leaf Photo)
- Interface: Telegram Bot API (`bot.py`)
- Preprocessing: Image Resizing & Normalization (`224x224`)
- Inference Engine: MobileNetV2 CNN Classifier (`inference.py`)
- Output: Crop & Disease Diagnosis + Confidence Score + Actionable Steps

## Key Features
- Lightweight CNN Transfer Learning: Fine-tuned MobileNetV2 architecture achieving 93.0% validation accuracy across key regional crops (tomato, grape, wheat).
- Field-Ready Augmentations: Trained with random rotations, horizontal flips, and color jitter to handle unconstrained outdoor lighting conditions.
- Low-Bandwidth Telegram Interface: Bilingual (Uzbek / Russian) conversational flow returning crop type, detected pathology, confidence score, and step-by-step agronomic recommendations.

## Quickstart

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model (optional):
   Place your dataset in `data/plant_village/train` and `data/plant_village/val`, then run:
   python train_model.py

3. Run the Telegram bot:
   export TELEGRAM_BOT_TOKEN="your-token-here"
   python bot.py

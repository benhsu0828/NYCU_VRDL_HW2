# Visual Recognition using Deep Learning 2026 Spring, Homework 2


Student Name: HSU, PAO-HUA
Student ID: 314581025

## Introduction

This homework includes two Python scripts:
- `data_augmentation.py`: offline augmentation with 1 image -> 3 images.
- `main.py`: two-stage training/validation with optional inference.

## Conda Environment

```bash
conda create -n nycu-cv-hw2 python=3.10 -y
conda activate nycu-cv-hw2
pip install torch torchvision transformers accelerate tqdm pillow
```
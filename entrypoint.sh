#!/bin/bash

# Train the model before running inference
echo "Training the model..."
python src/train.py

# Run inference after training
echo "Starting inference..."
python src/inference.py

"""Test configuration for training pipeline tests."""

import os
import sys

# Add the training directory to sys.path so we can import the scripts
training_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if training_dir not in sys.path:
    sys.path.insert(0, training_dir)

import pathlib
import sys

# Patch PosixPath to WindowsPath so pickle loads on Windows
pathlib.PosixPath = pathlib.WindowsPath

from fastai.vision.all import load_learner

model_path = 'model/my_custom_cnn_windows.pkl'

learn = load_learner(model_path, cpu=True)
print("Model loaded successfully!")


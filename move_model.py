import pickle
from pathlib import WindowsPath, PosixPath

model_path = 'model/my_custom_cnn.pkl'

# Load the raw pickle bytes
with open(model_path, 'rb') as f:
    raw_bytes = f.read()

# Replace PosixPath with WindowsPath in bytes
raw_bytes = raw_bytes.replace(b'PosixPath', b'WindowsPath')

# Write back the patched model file (you may want to backup first)
patched_model_path = 'model/my_custom_cnn_windows.pkl'
with open(patched_model_path, 'wb') as f:
    f.write(raw_bytes)

print(f"Patched model saved to {patched_model_path}")


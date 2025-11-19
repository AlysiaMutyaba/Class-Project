import os
import torch
import traceback
from fastai.vision.all import PILImage, load_learner
import pathlib
import numpy as np

pathlib.PosixPath = pathlib.WindowsPath

print("="*60)
print("MODEL PREDICTION TEST - DIRECT INFERENCE")
print("="*60)

# Load model
model_path = os.path.join("model", "my_custom_cnn_windows.pkl")
print(f"\n[STEP 1] Loading model from: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")

try:
    learn = load_learner(model_path, cpu=True)
    print(f"✓ Model loaded successfully")
    print(f"Model type: {type(learn)}")
    print(f"Model vocab: {learn.dls.vocab}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# Test with sample image
test_image_path = "static/uploads/Sample_image.jpg"
print(f"\n[STEP 2] Loading test image: {test_image_path}")
print(f"Image exists: {os.path.exists(test_image_path)}")

if not os.path.exists(test_image_path):
    print(f"✗ Test image not found!")
    exit(1)

try:
    img = PILImage.create(test_image_path)
    print(f"✓ Image loaded successfully")
    print(f"Image size: {img.size}")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    traceback.print_exc()
    exit(1)

# Test prediction using RAW MODEL INFERENCE
print(f"\n[STEP 3] Running prediction using raw model inference...")
print(f"Setting model to eval mode...")
learn.model.eval()
print(f"Model is in eval mode: {not learn.model.training}")

try:
    print(f"Preprocessing image...")
    dls = learn.dls
    
    # FIX: Set num_workers to 0 for Windows
    print(f"Creating test dataloader with num_workers=0...")
    test_dl = dls.test_dl([img], num_workers=0)
    print(f"✓ Test dataloader created")
    
    print(f"[WAITING FOR PREDICTION...]")
    with torch.no_grad():
        # Get batch from dataloader
        print(f"Extracting batch...")
        batch = next(iter(test_dl))
        print(f"✓ Batch extracted: {len(batch)} items")
        print(f"Batch tensor shape: {batch[0].shape}")
        
        # Run model
        print(f"Running model forward pass...")
        preds = learn.model(batch[0])
        print(f"✓ Model inference complete!")
        print(f"Predictions shape: {preds.shape}")
        print(f"Raw predictions: {preds}")
        
        # Get probabilities
        probs = torch.softmax(preds, dim=1)
        print(f"Probabilities: {probs}")
        
        # Get class
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = learn.dls.vocab[pred_idx]
        max_prob = float(probs[0, pred_idx])
        
        print(f"\n✓ PREDICTION SUCCESSFUL!")
        print(f"Predicted class index: {pred_idx}")
        print(f"Predicted class: {pred_class}")
        print(f"Confidence: {max_prob*100:.2f}%")

except Exception as e:
    print(f"\n✗ PREDICTION FAILED!")
    print(f"Error: {e}")
    print(f"Error type: {type(e).__name__}")
    traceback.print_exc()
    exit(1)

print(f"\n" + "="*60)
print(f"TEST COMPLETED SUCCESSFULLY")
print(f"="*60)
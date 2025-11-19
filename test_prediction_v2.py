import os
import torch
import traceback
from fastai.vision.all import PILImage, load_learner
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

if __name__ == '__main__':
    print("="*60)
    print("MODEL PREDICTION TEST - MANUAL PREPROCESSING")
    print("="*60)

    # Load model
    model_path = os.path.join("model", "my_custom_cnn_windows.pkl")
    print(f"\n[STEP 1] Loading model from: {model_path}")

    try:
        learn = load_learner(model_path, cpu=True)
        print(f"✓ Model loaded successfully")
        print(f"Model vocab: {learn.dls.vocab}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        traceback.print_exc()
        exit(1)

    # Load test image
    test_image_path = "static/uploads/Sample_image.jpg"
    print(f"\n[STEP 2] Loading test image: {test_image_path}")

    if not os.path.exists(test_image_path):
        print(f"✗ Test image not found!")
        exit(1)

    try:
        img = PILImage.create(test_image_path)
        print(f"✓ Image loaded - Size: {img.size}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        traceback.print_exc()
        exit(1)

    # Prediction
    print(f"\n[STEP 3] Running prediction with manual preprocessing...")
    learn.model.eval()

    try:
        with torch.no_grad():
            print(f"Getting item transforms...")
            # Get the item transforms (this is what processes individual images)
            dls = learn.dls
            print(f"DLS type: {type(dls)}")
            print(f"Available attributes: {dir(dls)}")
            
            # Try to get transforms from the dataloader
            if hasattr(dls, 'after_item'):
                print(f"Using after_item transforms")
                item_tfms = dls.after_item
            elif hasattr(dls, 'train'):
                print(f"Using train transforms")
                item_tfms = dls.train.tfms
            else:
                print(f"Getting default transforms")
                item_tfms = None
            
            print(f"Item transforms: {item_tfms}")
            
            # Method 1: Try using the learner's predict method but catch the multiprocessing issue
            print(f"\nTrying alternative: Direct model forward pass...")
            
            # Resize image to model input size (typically 224x224 for vision models)
            img_resized = img.resize((224, 224))
            print(f"✓ Resized to: {img_resized.size}")
            
            # Convert PIL to tensor
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img_resized)
            print(f"✓ Converted to tensor - Shape: {img_tensor.shape}")
            
            # Normalize (ImageNet stats)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            img_normalized = normalize(img_tensor)
            print(f"✓ Normalized - Shape: {img_normalized.shape}")
            
            # Add batch dimension
            img_batch = img_normalized.unsqueeze(0)
            print(f"✓ Batch created - Shape: {img_batch.shape}")
            
            print(f"Running inference...")
            preds = learn.model(img_batch)
            print(f"✓ Inference complete!")
            print(f"Raw predictions shape: {preds.shape}")
            print(f"Raw predictions: {preds}")
            
            # Get probabilities
            probs = torch.softmax(preds, dim=1)
            print(f"Probabilities: {probs}")
            
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = learn.dls.vocab[pred_idx]
            max_prob = float(probs[0, pred_idx])

        print(f"\n✓ SUCCESS!")
        print(f"Predicted class: '{pred_class}'")
        print(f"Predicted index: {pred_idx}")
        print(f"Confidence: {max_prob*100:.2f}%")
        print(f"\n" + "="*60)

    except Exception as e:
        print(f"\n✗ FAILED!")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        exit(1)
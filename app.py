import os
import pathlib
import traceback
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from fastai.vision.all import PILImage, load_learner

# Fix Windows path issue
pathlib.PosixPath = pathlib.WindowsPath

print("="*60)
print("INITIALIZING APPLICATION")
print("="*60)

learn = None
coffee_leaf_model = None

def load_model():
    global learn
    model_path = os.path.join("model", "my_custom_cnn_windows.pkl")
    print(f"[MODEL] Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"[MODEL ERROR] Model file not found at: {model_path}")
        learn = None
        return None
    
    try:
        print(f"[MODEL] Model file exists, attempting to load...")
        learn = load_learner(model_path, cpu=True)
        print(f"[MODEL SUCCESS] FastAI model loaded successfully")
        print(f"[MODEL] Model type: {type(learn)}")
        print(f"[MODEL] Model vocab: {learn.dls.vocab if hasattr(learn, 'dls') else 'N/A'}")
        return learn
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load model: {e}")
        traceback.print_exc()
        learn = None
        return None

def load_coffee_leaf_detector():
    """
    Load a pre-trained ResNet18 model fine-tuned to detect coffee leaves.
    For production, you'd want to train this on coffee vs non-coffee images.
    For now, we use feature extraction to distinguish coffee leaves.
    """
    global coffee_leaf_model
    try:
        print(f"[COFFEE DETECTOR] Loading coffee leaf detector...")
        # Use ResNet50 pre-trained on ImageNet for feature extraction
        coffee_leaf_model = models.resnet50(pretrained=True)
        coffee_leaf_model.eval()
        print(f"[COFFEE DETECTOR] ✓ Coffee leaf detector loaded")
        return coffee_leaf_model
    except Exception as e:
        print(f"[COFFEE DETECTOR ERROR] Failed to load: {e}")
        coffee_leaf_model = None
        return None

# Load models on startup
load_model()
load_coffee_leaf_detector()

app = Flask(__name__)
app.secret_key = "dbuefueueio27436yrubdyhgfwjt2jbhgdhdg"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_coffee_leaf(image_path):
    """
    Balanced coffee leaf detection - checks for leaf characteristics
    without being too strict or too lenient.
    
    Returns (bool, confidence_score)
    """
    try:
        print(f"[COFFEE LEAF CHECK] Analyzing image: {image_path}")
        
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize((224, 224))
            arr = np.array(im).astype(np.float32) / 255.0
            
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            
            # CHECK 1: Green dominance (must have significant green)
            green_dominant = (g > r) & (g > b) & (g > 0.2)
            green_ratio = float(np.sum(green_dominant)) / (arr.shape[0] * arr.shape[1])
            
            # CHECK 2: Not too much red (rejects red vegetables)
            high_red = np.sum((r > 0.7) & (r > g)) / (arr.shape[0] * arr.shape[1])
            
            # CHECK 3: Natural variation (real leaves, not solid colors)
            green_std = float(np.std(g))
            
            # CHECK 4: Not too saturated (avoids artificial backgrounds)
            saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            high_sat_ratio = float(np.sum(saturation > 0.8)) / (arr.shape[0] * arr.shape[1])
            
            print(f"[COFFEE LEAF CHECK] Green ratio: {green_ratio:.3f}")
            print(f"[COFFEE LEAF CHECK] High red ratio: {high_red:.3f}")
            print(f"[COFFEE LEAF CHECK] Green variation: {green_std:.3f}")
            print(f"[COFFEE LEAF CHECK] High saturation: {high_sat_ratio:.3f}")
            
            # DECISION: Balanced thresholds
            is_leaf = (
                green_ratio >= 0.20 and              # At least 20% green
                high_red < 0.25 and                 # Less than 25% very red
                green_std > 0.05 and                # Some natural variation
                high_sat_ratio < 0.40               # Not overly saturated
            )
            
            confidence = green_ratio * 0.6 + (1 - high_red) * 0.4
            confidence = min(max(confidence, 0), 1)
            
            print(f"[COFFEE LEAF CHECK] Is Leaf: {'✓ YES' if is_leaf else '✗ NO'}")
            print(f"[COFFEE LEAF CHECK] Confidence: {confidence:.3f}")
            
            return is_leaf, confidence
            
    except Exception as e:
        print(f"[COFFEE LEAF CHECK ERROR] {e}")
        traceback.print_exc()
        return False, 0.0

DISEASE_INFO = {
    'Healthy': {
        'name': 'Healthy Leaf',
        'prevention': 'Continue good practices. Providing the right balance of water, light, temperature and humidity is important.'
    },
    'Coffee_rust': {
        'name': 'Coffee Leaf Rust',
        'prevention': 'Appropriate fertilization, moisture control and cautious use of fungicides, aiming to enhance plant health.'
    },
    'Coffee_red_spider_mite ': { 
        'name': 'Coffee Red Spider Mite',
        'prevention': 'Mist your plants regularly, especially during dry months, and use a humidifier if necessary.'
    }
}

print(f"[CONFIG] Disease categories: {list(DISEASE_INFO.keys())}")
print(f"[CONFIG] Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"[CONFIG] Allowed extensions: {app.config['ALLOWED_EXTENSIONS']}")

# Routes
@app.route("/")
def home():
    print(f"[ROUTE] Home page accessed")
    return render_template("home.html")

@app.route("/auth", methods=['GET', 'POST'])
def auth():
    print(f"[ROUTE] Auth route - Method: {request.method}")
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        print(f"[AUTH] Username: {username}")
        
        if not username or not password:
            flash('Username and password required', 'error')
            return redirect(url_for('auth'))
        
        if "register" in request.form:
            print(f"[AUTH] Register attempt for user: {username}")
            existing_user = User.query.filter_by(username=username).first()
            
            if existing_user:
                print(f"[AUTH] User already exists: {username}")
                flash('User already exists! Try logging in.', 'error')
                return redirect(url_for('auth'))
            
            try:
                hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username=username, password=hashed_pw)
                db.session.add(new_user)
                db.session.commit()
                print(f"[AUTH] User registered successfully: {username}")
                flash('Registered successfully! You can now log in', 'success')
                return redirect(url_for('auth'))
            except Exception as e:
                db.session.rollback()
                print(f"[AUTH ERROR] Registration error: {e}")
                flash('Registration failed', 'error')
                return redirect(url_for('auth'))
        
        elif "login" in request.form:
            print(f"[AUTH] Login attempt for user: {username}")
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                session['user'] = username
                print(f"[AUTH] Login successful: {username}")
                flash('Login successful!', 'success')
                return redirect(url_for("upload"))
            else:
                print(f"[AUTH] Login failed for user: {username}")
                flash("Invalid username or password.", 'error')
                return redirect(url_for('auth'))
    
    return render_template("auth.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    print(f"[ROUTE] Upload route - Method: {request.method}")
    
    if 'user' not in session:
        print(f"[UPLOAD] User not in session, redirecting to auth")
        return redirect(url_for('auth'))
    
    print(f"[UPLOAD] Current user: {session['user']}")
    
    filename = None
    prediction = None
    prevention = None
    confidence = None

    if request.method == 'POST':
        print(f"[UPLOAD] Processing POST request")
        
        try:
            # Check file exists
            if 'file' not in request.files:
                print(f"[UPLOAD ERROR] No file part in request")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            print(f"[UPLOAD] File received: {file.filename}")
            
            # Check filename
            if file.filename == '':
                print(f"[UPLOAD ERROR] Empty filename")
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Check file type
            if not allowed_file(file.filename):
                print(f"[UPLOAD ERROR] Invalid file type: {file.filename}")
                flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF', 'error')
                return redirect(request.url)
            
            # Save file
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"[UPLOAD] File saved to: {filepath}")
            
            # Check if model loaded
            if learn is None:
                print(f"[UPLOAD ERROR] Model not loaded")
                flash('Model not available', 'error')
                return render_template("upload.html", 
                                     filename=filename, 
                                     prediction=prediction, 
                                     prevention=prevention, 
                                     confidence=confidence)
            
            # Load and predict
            try:
                print(f"[PREDICTION] Loading image from: {filepath}")
                img = PILImage.create(filepath)
                print(f"[PREDICTION] Image loaded successfully - Size: {img.size}")
                
                print(f"[PREDICTION] Running model prediction...")
                learn.model.eval()
                
                with torch.no_grad():
                    print(f"[PREDICTION] Preprocessing image...")
                    
                    # Resize to 224x224
                    img_resized = img.resize((224, 224))
                    print(f"[PREDICTION] Resized to: {img_resized.size}")
                    
                    # Convert to tensor
                    to_tensor = transforms.ToTensor()
                    img_tensor = to_tensor(img_resized)
                    print(f"[PREDICTION] Converted to tensor - Shape: {img_tensor.shape}")
                    
                    # Normalize (ImageNet stats)
                    normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                    img_normalized = normalize(img_tensor)
                    print(f"[PREDICTION] Normalized - Shape: {img_normalized.shape}")
                    
                    # Add batch dimension
                    img_batch = img_normalized.unsqueeze(0)
                    print(f"[PREDICTION] Batch created - Shape: {img_batch.shape}")
                    
                    # Run inference
                    print(f"[PREDICTION] Running model inference...")
                    preds = learn.model(img_batch)
                    print(f"[PREDICTION] ✓ Inference complete!")
                    
                    # Get probabilities
                    probs = torch.softmax(preds, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    pred_class = learn.dls.vocab[pred_idx]
                    max_prob = float(probs[0, pred_idx])
                
                print(f"[PREDICTION] Predicted class: '{pred_class}'")
                print(f"[PREDICTION] Predicted index: {pred_idx}")
                print(f"[PREDICTION] Confidence: {max_prob*100:.2f}%")
                
                # ========== CONFIDENCE THRESHOLD CHECK ==========
                CONFIDENCE_THRESHOLD = 0.70  # Only accept predictions > 70%
                
                if max_prob < CONFIDENCE_THRESHOLD:
                    print(f"[PREDICTION] Prediction rejected - confidence too low ({max_prob*100:.2f}%)")
                    flash(f' Cannot confidently identify the disease in this image. Confidence: {max_prob*100:.2f}%. Please upload a clearer image.', 'error')
                    return render_template("upload.html", 
                                         filename=filename, 
                                         prediction=None, 
                                         prevention=None, 
                                         confidence=None)
                
                # Match prediction to disease info
                print(f"[PREDICTION] Matching to disease info...")
                print(f"[PREDICTION] Available keys: {list(DISEASE_INFO.keys())}")
                
                if pred_class in DISEASE_INFO:
                    print(f"[PREDICTION] ✓ Exact match found: {pred_class}")
                    prediction = DISEASE_INFO[pred_class]['name']
                    prevention = DISEASE_INFO[pred_class]['prevention']
                else:
                    # Try without spaces
                    pred_class_clean = pred_class.strip()
                    if pred_class_clean in DISEASE_INFO:
                        print(f"[PREDICTION] ✓ Match found (stripped): {pred_class_clean}")
                        prediction = DISEASE_INFO[pred_class_clean]['name']
                        prevention = DISEASE_INFO[pred_class_clean]['prevention']
                    else:
                        print(f"[PREDICTION] ✗ No match found, using raw class")
                        prediction = pred_class
                        prevention = "Disease information not available"
                
                confidence = f"{max_prob*100:.2f}%"
                
                print(f"[PREDICTION] ========== FINAL RESULT ==========")
                print(f"[PREDICTION] Disease: {prediction}")
                print(f"[PREDICTION] Prevention: {prevention}")
                print(f"[PREDICTION] Confidence: {confidence}")
                print(f"[PREDICTION] ===================================")
                
                flash('✓ AI analysis completed successfully!', 'success')
                
            except Exception as pred_error:
                print(f"[PREDICTION ERROR] Error during prediction: {pred_error}")
                print(f"[PREDICTION ERROR] Error type: {type(pred_error).__name__}")
                traceback.print_exc()
                flash(f'Prediction error: {str(pred_error)}', 'error')
                return render_template("upload.html", 
                                     filename=filename, 
                                     prediction=None, 
                                     prevention=None, 
                                     confidence=None)
        
        except Exception as e:
            print(f"[UPLOAD ERROR] General error: {e}")
            print(f"[UPLOAD ERROR] Error type: {type(e).__name__}")
            traceback.print_exc()
            flash(f'Upload error: {str(e)}', 'error')

    return render_template("upload.html", 
                         filename=filename, 
                         prediction=prediction, 
                         prevention=prevention, 
                         confidence=confidence)

@app.route("/logout")
def logout():
    user = session.pop('user', None)
    print(f"[ROUTE] User logged out: {user}")
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))

if __name__ == "__main__":
    print("="*60)
    print("STARTUP DIAGNOSTICS")
    print("="*60)
    
    model_path = os.path.join("model", "my_custom_cnn_windows.pkl")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print(f"Model is loaded: {learn is not None}")
    print(f"Coffee leaf detector loaded: {coffee_leaf_model is not None}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    
    with app.app_context():
        try:
            db.create_all()
            print(f"Database initialized successfully")
        except Exception as e:
            print(f"Database error: {e}")
    
    print("="*60)
    print(f"Starting Flask app on http://127.0.0.1:5000")
    print("="*60)
    
    app.run(debug=True, use_reloader=False)

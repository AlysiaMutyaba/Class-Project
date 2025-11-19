import os
import pathlib
import traceback
import sys
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

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

# Load model on startup
load_model()

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

DISEASE_INFO = {
    'Healthy': {
        'name': 'Healthy Leaf',
        'prevention': 'Continue good practices'
    },
    'Coffee_rust': {
        'name': 'Coffee Leaf Rust',
        'prevention': 'Apply fungicides regularly'
    },
    'Coffee_red_spider_mite ': {  # NOTE: Space at end to match vocab!
        'name': 'Coffee Red Spider Mite',
        'prevention': 'Apply acaricide and maintain humidity'
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
                return render_template("upload.html", filename=filename, prediction=prediction, prevention=prevention, confidence=confidence)
            
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
                return render_template("upload.html", filename=filename, prediction=None, prevention=None, confidence=None)
        
        except Exception as e:
            print(f"[UPLOAD ERROR] General error: {e}")
            print(f"[UPLOAD ERROR] Error type: {type(e).__name__}")
            traceback.print_exc()
            flash(f'Upload error: {str(e)}', 'error')

    return render_template("upload.html", filename=filename, prediction=prediction, prevention=prevention, confidence=confidence)

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

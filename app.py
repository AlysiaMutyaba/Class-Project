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
        learn = load_learner(model_path, cpu=True)
        print(f"[MODEL SUCCESS] FastAI model loaded successfully")
        print(f"[MODEL] Model vocab: {learn.dls.vocab if hasattr(learn, 'dls') else 'N/A'}")
        return learn
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load model: {e}")
        traceback.print_exc()
        learn = None
        return None

def load_coffee_leaf_detector():
    """
    Load a pre-trained ResNet model for general feature extraction.
    """
    global coffee_leaf_model
    try:
        print(f"[COFFEE DETECTOR] Loading coffee leaf detector...")
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


# ============================
#   COFFEE LEAF DETECTOR
# ============================
def is_coffee_leaf(image_path):
    """
    Detects if an image is a coffee leaf using color heuristics.
    Returns (bool, confidence)
    """
    try:
        print(f"[COFFEE LEAF CHECK] Analyzing image: {image_path}")
        
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize((224, 224))
            arr = np.array(im).astype(np.float32) / 255.0
            
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            
            # Green dominance
            green_dominant = (g > r) & (g > b) & (g > 0.2)
            green_ratio = float(np.sum(green_dominant)) / (arr.shape[0] * arr.shape[1])
            
            # Red exclusion
            high_red = np.sum((r > 0.7) & (r > g)) / (arr.shape[0] * arr.shape[1])
            
            # Leaf variation
            green_std = float(np.std(g))
            
            # Saturation
            saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
            high_sat_ratio = float(np.sum(saturation > 0.8)) / (arr.shape[0] * arr.shape[1])
            
            print(f"[COFFEE LEAF CHECK] Green ratio: {green_ratio:.3f}")
            print(f"[COFFEE LEAF CHECK] High red ratio: {high_red:.3f}")
            print(f"[COFFEE LEAF CHECK] Green variation: {green_std:.3f}")
            print(f"[COFFEE LEAF CHECK] High saturation: {high_sat_ratio:.3f}")
            
            is_leaf = (
                green_ratio >= 0.20 and
                high_red < 0.25 and
                green_std > 0.05 and
                high_sat_ratio < 0.40
            )
            
            confidence = green_ratio * 0.6 + (1 - high_red) * 0.4
            confidence = min(max(confidence, 0), 1)
            
            print(f"[COFFEE LEAF CHECK] Is Leaf: {is_leaf}")
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
        'prevention': 'Appropriate fertilization, moisture control and cautious use of fungicides.'
    },
    'Coffee_red_spider_mite': { 
        'name': 'Coffee Red Spider Mite',
        'prevention': 'Mist plants regularly and increase humidity.'
    }
}

# ============================
#          ROUTES
# ============================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/auth", methods=['GET', 'POST'])
def auth():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash('Username and password required', 'error')
            return redirect(url_for('auth'))
        
        if "register" in request.form:
            existing_user = User.query.filter_by(username=username).first()
            
            if existing_user:
                flash('User already exists! Try logging in.', 'error')
                return redirect(url_for('auth'))
            
            try:
                hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
                new_user = User(username=username, password=hashed_pw)
                db.session.add(new_user)
                db.session.commit()
                flash('Registered successfully! You can now log in', 'success')
                return redirect(url_for('auth'))
            except:
                db.session.rollback()
                flash('Registration failed', 'error')
                return redirect(url_for('auth'))
        
        elif "login" in request.form:
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                session['user'] = username
                flash('Login successful!', 'success')
                return redirect(url_for("upload"))
            else:
                flash("Invalid username or password.", 'error')
                return redirect(url_for('auth'))
    
    return render_template("auth.html")


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('auth'))
    
    filename = None
    prediction = None
    prevention = None
    confidence = None

    if request.method == 'POST':
        
        file = request.files.get('file')

        if not file or file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Upload PNG, JPG, JPEG, or GIF', 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # =============================
        #  COFFEE LEAF VERIFICATION
        # =============================
        print("[CHECK] Verifying if coffee leaf...")
        is_leaf, leaf_conf = is_coffee_leaf(filepath)

        MIN_COFFEE_CONFIDENCE = 0.30

        if not is_leaf or leaf_conf < MIN_COFFEE_CONFIDENCE:
            flash("This image does not appear to be a coffee leaf. Please upload a clear coffee leaf photo.", 'error')
            return render_template("upload.html", filename=filename)

        print("[CHECK RESULT] ✓ Image confirmed as coffee leaf")

        # =============================
        #        PREDICTION
        # =============================
        if learn is None:
            flash('Model not available', 'error')
            return render_template("upload.html", filename=filename)

        try:
            img = PILImage.create(filepath)
            img_resized = img.resize((224, 224))
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img_resized)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            img_normalized = normalize(img_tensor)
            img_batch = img_normalized.unsqueeze(0)

            preds = learn.model(img_batch)
            probs = torch.softmax(preds, dim=1)

            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = learn.dls.vocab[pred_idx]
            max_prob = float(probs[0, pred_idx])

            CONFIDENCE_THRESHOLD = 0.70

            if max_prob < CONFIDENCE_THRESHOLD:
                flash(f"Low confidence ({max_prob*100:.2f}%). Upload a clearer image.", 'error')
                return render_template("upload.html", filename=filename)

            # Match disease info
            pred_clean = pred_class.strip()
            if pred_clean in DISEASE_INFO:
                prediction = DISEASE_INFO[pred_clean]['name']
                prevention = DISEASE_INFO[pred_clean]['prevention']
            else:
                prediction = pred_clean
                prevention = "Disease info not available"

            confidence = f"{max_prob*100:.2f}%"
            flash("Analysis completed successfully!", 'success')

        except Exception as e:
            flash(f"Prediction error: {str(e)}", 'error')

    return render_template(
        "upload.html",
        filename=filename,
        prediction=prediction,
        prevention=prevention,
        confidence=confidence
    )


@app.route("/logout")
def logout():
    session.pop('user', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('home'))


if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
            print("Database initialized successfully")
        except Exception as e:
            print(f"Database error: {e}")
    
    print("Starting Flask app...")
    app.run(debug=True, use_reloader=False)
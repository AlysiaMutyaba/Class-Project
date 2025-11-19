from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from fastai.vision.all import PILImage
import torch

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from fastai.vision.all import load_learner

learn = None
def load_model():
    global learn
    model_path = os.path.join("model","my_custom_cnn_windows.pkl")
    print(f"Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print("Debug: model file not found at this path")
        learn = None
        return None
        
    try:
        learn = load_learner(model_path, cpu=True)
        print("FastAI model loaded successfully")
        return learn
    except Exception as e:
        print("Error loading FastAI model:",e)
        learn = None
        return None
load_model()

app = Flask(__name__)
app.secret_key = "dbuefueueio27436yrubdyhgfwjt2jbhgdhdg"
app.config["SQLALCHEMY_DATABASE_URI"]= "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
db= SQLAlchemy(app)

#User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
 #functions
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path, target_size=(224,224)):
    print(f"Debug: preprocess_image called with: {image_path}")
    print(f"Debug: File exists: {os.path.exists(image_path)}")

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found : {image_path}")
        
        print(f"Debug: Opening image...")
        img = Image.open(image_path)
        print(f"Debug: Image format: {img.format}, Size:{img.size}, Mode: {img.mode}")

        print(f"Debug: Resizing image to {target_size}...")
        img = img.resize(target_size)

        print(f"Debug: Conerting to array")
        img_array = np.array(img) / 255.0
        print(f"Debug: Array shape: {img_array.shape},Dtype:{img_array.dtype}")

        print(f"Debug: Expanding dimensions...")
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Debug: Final array shape: {img_array.shape}")

        return img_array
    
    except Exception as e:
        print(f"Debug: Error in preprocess_image: {e}")
        import traceback
        traceback.print_exc()
        raise

DISEASE_INFO = {
    'Healthy':{'name':'Healthy Leaf', 'prevention': 'Continue good practices'},
    'Coffee_rust' : {'name':'Coffee Leaf Rust', 'prevention':'Apply fungicides'},
    'Coffee_red_spider_mite ': {'name' : 'Coffee Red Spider Mite', 'prevention':'Apply fungicide'}
}


#routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/auth", methods=['GET','POST'])
def auth():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if "register" in  request.form:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('User already exists! Try logging in.', 'error')
                return redirect(url_for('auth'))
            hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_pw)
            db.session.add(new_user)
            db.session.commit()
            flash('Registered successfully! You can now log in','success')
            return redirect(url_for('auth'))  
        elif "login" in request.form:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session['user'] = username
                return redirect(url_for("upload"))   
            else:
                flash("Invalid username or password.", 'error')
                return redirect(url_for('auth'))             
    return render_template("auth.html")

@app.route("/upload", methods=['GET','POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('auth'))
    filename = None
    prediction = None
    prevention = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)
            print(f"Debug: File saved to: {filepath}")

            try:
                if learn is not None:
                    print("Debug: Calling FastAI model to predict..")
                    img = PILImage.create(filepath)
                    print("Image loaded")
                    with torch.no_grad():
                        pred_class, pred_idx, probs = learn.predict(img)
                        print("Predicting")
                    predicted_disease = str(pred_class)
                    print(f"Predicted class: {predicted_disease}")

                    if predicted_disease in DISEASE_INFO:
                        prediction = DISEASE_INFO[predicted_disease]['name']
                        prevention = DISEASE_INFO[predicted_disease]['prevention']
                    else:
                        prediction = predicted_disease
                        prevention = "No preventation information available"

                    flash('AI analysis completed successfully!', 'success')
                else:
                    print("Debug: Model is none")
                    flash('Model not available','error')
            except Exception as e:
                print(f"Debug: Error in upload route:{e}")
                flash(f'Error during AI analysis: {str(e)}', 'error')

    return render_template("upload.html", filename=filename, prediction=prediction, prevention=prevention)
    
def transform_image_for_prediction(image):
    """
    load the image
    imag = use cv2 to read
    transform imageusing cv2 blur
    preprocess it
    and then put it through the model
    """
    transformed_image = ''
    return transformed_image




@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("auth"))

if __name__ == "__main__":
    print("STARTUP CHECK:")
    model_path = os.path.join("model", "my_custom_cnn.pkl")
    print("Model path:",model_path)
    print("Model exists:",os.path.exists(model_path))

    with app.app_context():
        db.create_all()
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            print(f"Created upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True)

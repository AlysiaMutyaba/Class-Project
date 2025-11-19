# AgroPredict - AI Coffee Leaf Disease Detection System

## Project Overview

AgroPredict is a web-based application that uses artificial intelligence and deep learning to automatically detect and classify diseases in coffee leaves. The system helps farmers identify coffee leaf rust, spider mite infestations, and healthy leaves with high accuracy.

### Key Features

- User Authentication (Registration and Login)
- Real-time Image Upload and Processing
- AI-Powered Disease Classification
- Disease Prevention Recommendations
- Confidence Score Display
- Responsive Web Interface
- Secure Database Management

### Supported Diseases

1. **Healthy Leaf** - No disease detected
2. **Coffee Rust (Hemileia vastatrix)** - Fungal disease causing reddish-brown lesions
3. **Coffee Red Spider Mite (Oligonychus yothersi)** - Pest causing yellowing and leaf drop

---

## System Architecture

```
AgroPredict/
├── app.py                          # Flask application server
├── requirements.txt                # Python dependencies
├── users.db                        # SQLite database
├── model/
│   └── my_custom_cnn_windows.pkl   # Trained FastAI model
├── static/
│   ├── CSS/
│   │   ├── home.css
│   │   ├── auth.css
│   │   └── upload.css
│   ├── images/
│   │   ├── coffee.jpg              # Homepage image
│   │   ├── chip.png                # Feature icon
│   │   ├── bk_2.jpg                # Login background
│   │   └── bk1_image.jpg           # Upload background
│   └── uploads/                    # User uploaded images
├── templates/
│   ├── home.html
│   ├── auth.html
│   └── upload.html
├── docs/
│   ├── INSTALLATION.md
│   ├── API_REFERENCE.md
│   ├── USER_GUIDE.md
│   └── images/
│       ├── homepage.png
│       ├── login.png
│       ├── upload.png
│       └── prediction.png
└── test_prediction_v2.py           # Model testing script
```

---

## Installation Guide

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- 2GB RAM minimum
- 500MB disk space

### Step 1: Clone or Download Project

```bash
cd c:\Users\YourUsername\Desktop
git clone <repository-url>
cd Class-Project
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python test_prediction_v2.py
```

You should see:
```
SUCCESS!
Predicted class: 'Healthy'
Predicted index: 2
Confidence: 99.87%
```

### Step 5: Run Application

```bash
python app.py
```

Access the application at: `http://127.0.0.1:5000`

---

## User Guide

### Registration

1. Navigate to http://127.0.0.1:5000
2. Click "Go to Login"
3. Enter desired username and password
4. Click "Register"
5. Login with your credentials

### Uploading an Image

1. Login to your account
2. Navigate to Upload page
3. Click file input or drag-and-drop an image
4. Supported formats: PNG, JPG, JPEG, GIF
5. Maximum file size: 16MB
6. Click "Upload Image"

### Viewing Results

After upload, results appear in a modal window showing:

- **Analyzed Image** - Your uploaded coffee leaf photo
- **Disease Classification** - Identified disease status
- **Confidence Score** - Prediction accuracy percentage
- **Confidence Bar** - Visual confidence indicator
- **Prevention Recommendations** - Actionable advice

### Actions

- **Upload Another** - Upload and analyze a different image
- **Home** - Return to homepage
- **Logout** - Safely exit your account

---

## API Routes

### Authentication Routes

#### POST /auth
User registration and login

**Request:**
```json
{
  "username": "string",
  "password": "string",
  "register": "true/false"
}
```

**Response:**
- Success: Redirect to /upload or /auth
- Error: Redirect to /auth with error message

#### GET /logout
Logout current user session

**Response:**
- Redirect to homepage

### Image Upload Routes

#### GET /upload
Display upload form

**Response:**
- HTML upload page (requires authentication)

#### POST /upload
Upload and analyze image

**Request:**
- File: multipart/form-data
- Field name: "file"

**Response:**
- HTML page with results modal containing:
  - `filename`: string
  - `prediction`: string (disease name)
  - `confidence`: string (percentage)
  - `prevention`: string (recommendations)

### Home Routes

#### GET /
Display homepage

**Response:**
- HTML homepage

---

## Model Information

### Model Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Framework:** FastAI Vision
- **Backend:** PyTorch
- **Input Size:** 224x224 pixels
- **Output Classes:** 3 (Healthy, Coffee Rust, Red Spider Mite)
- **File:** my_custom_cnn_windows.pkl

### Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~99%+ |
| Input Formats | PNG, JPG, JPEG, GIF |
| Processing Time | 1-3 seconds |
| Confidence Range | 0-100% |

### Image Preprocessing Pipeline

1. **Load** - PIL image loading
2. **Resize** - 224x224 with crop method
3. **Convert** - PIL to tensor
4. **Normalize** - ImageNet normalization
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
5. **Batch** - Add batch dimension
6. **Inference** - Model forward pass

---

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(200) NOT NULL
);
```

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY |
| username | VARCHAR(100) | UNIQUE, NOT NULL |
| password | VARCHAR(200) | NOT NULL |

---

## Troubleshooting

### Issue: Model Loading Error

**Error:** `[MODEL ERROR] Failed to load model`

**Solution:**
1. Verify model file exists at: `model/my_custom_cnn_windows.pkl`
2. Check file permissions
3. Ensure sufficient disk space
4. Reinstall FastAI: `pip install --upgrade fastai`

### Issue: Prediction Hanging

**Error:** Process hangs at prediction step

**Solution:**
1. Ensure Python is running from virtual environment
2. Check system RAM availability
3. Restart Flask application
4. Verify image format (must be PNG, JPG, JPEG, or GIF)

### Issue: Database Error

**Error:** `[Database error]`

**Solution:**
1. Delete `users.db` file
2. Restart application (database will be recreated)
3. Register new account

### Issue: Port Already in Use

**Error:** `Address already in use` on port 5000

**Solution:**
1. Stop other Flask instances
2. Use different port: `app.run(debug=True, port=5001)`
3. Or kill process: `netstat -ano | findstr :5000`

### Issue: File Upload Fails

**Error:** File upload not working

**Solution:**
1. Check file size (max 16MB)
2. Verify file format (PNG, JPG, JPEG, GIF only)
3. Ensure write permissions in `static/uploads/` directory
4. Clear uploads folder if full

---

## Security Considerations

### Password Security

- Passwords hashed with PBKDF2-SHA256
- Never stored in plain text
- Never transmitted unencrypted

### Session Management

- Session tokens stored in cookies
- Auto-logout on browser close
- Session validation on each request

### File Upload Security

- File type validation (whitelist: PNG, JPG, JPEG, GIF)
- Filename sanitization with `secure_filename()`
- Maximum file size: 16MB
- Files stored in isolated directory

### Database Security

- SQLite with parameterized queries
- SQLAlchemy ORM prevents SQL injection
- No raw SQL queries

---

## Performance Optimization

### Model Performance

- CPU inference: 1-3 seconds
- Memory usage: ~500MB
- No GPU required

### Application Performance

- Asynchronous file operations
- Efficient image preprocessing
- Optimized database queries
- Static file caching

### Recommendations

- Use SSD for faster file I/O
- Allocate at least 2GB RAM
- Run on modern processor (i5/Ryzen 5+)
- Use wired internet connection

---

## Development Guide

### Adding New Disease Classes

1. **Retrain Model**
   - Collect images of new disease
   - Use FastAI training pipeline
   - Save as `my_custom_cnn_windows.pkl`

2. **Update DISEASE_INFO Dictionary** (app.py)
   ```python
   DISEASE_INFO = {
       'New_Disease': {
           'name': 'Disease Display Name',
           'prevention': 'Prevention advice'
       }
   }
   ```

3. **Test Prediction**
   ```bash
   python test_prediction_v2.py
   ```

### Modifying UI

- **Styles:** Edit CSS files in `static/CSS/`
- **Templates:** Edit HTML files in `templates/`
- **Colors:** Update gradient values in stylesheets
- **Fonts:** Modify font-family in base styles

### Database Modifications

1. Update User model in app.py
2. Delete users.db to reset
3. Restart application
4. Test new schema

---

## Deployment Guide

### Local Network

```bash
app.run(host='0.0.0.0', port=5000, debug=False)
```

### Production Deployment

Use Gunicorn WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Cloud Deployment

1. Push code to GitHub
2. Connect to cloud provider (Heroku, AWS, Azure)
3. Set environment variables
4. Deploy with provider's CLI

---

## Testing

### Unit Testing

```bash
python test_prediction_v2.py
```

### Manual Testing
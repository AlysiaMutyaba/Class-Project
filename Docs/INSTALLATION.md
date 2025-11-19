# Installation Guide - AgroPredict

## Quick Start (5 minutes)

### Windows Installation

```bash
# 1. Open PowerShell and navigate to project
cd "C:\Users\YourUsername\Desktop\Class-Project"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python app.py

# 6. Open browser
# Visit: http://127.0.0.1:5000
```

### macOS/Linux Installation

```bash
# 1. Navigate to project
cd ~/Desktop/Class-Project

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python app.py

# 6. Open browser
# Visit: http://127.0.0.1:5000
```

## Detailed Installation

### Step 1: Verify Python

```bash
python --version
# Should output: Python 3.10.x or higher
```

### Step 2: Clone Repository

```bash
git clone <repository-url>
cd Class-Project
```

### Step 3: Virtual Environment Setup

```bash
# Create
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Verify activation (should show (.venv) prefix)
python -c "import sys; print(sys.executable)"
```

### Step 4: Install Requirements

```bash
pip install -r requirements.txt
```

Installation should complete without errors. If issues occur, see Troubleshooting.

### Step 5: Verify Installation

```bash
# Run model test
python test_prediction_v2.py

# Expected output:
# SUCCESS!
# Predicted class: 'Healthy'
# Confidence: 99.87%
```

### Step 6: Run Application

```bash
python app.py
```

Expected output:
```
INITIALIZING APPLICATION
...
Starting Flask app on http://127.0.0.1:5000
```

### Step 7: Access Application

Open web browser and visit:
```
http://127.0.0.1:5000
```

## Troubleshooting Installation

### Python Not Found

**Error:** `'python' is not recognized`

**Solution:**
```bash
# Try python3
python3 --version

# Or add Python to PATH
# Windows: System Properties > Environment Variables > Add Python path
```

### Permission Denied

**Error:** `.venv\Scripts\Activate.ps1 cannot be loaded`

**Solution (Windows PowerShell):**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Module Import Errors

**Error:** `ModuleNotFoundError: No module named 'fastai'`

**Solution:**
```bash
# Ensure virtual environment is activated
# Then reinstall
pip install --upgrade fastai torch
```

### Disk Space Issues

**Error:** `No space left on device`

**Solution:**
```bash
# Check disk space
df -h

# Clean pip cache
pip cache purge

# Or use less storage:
pip install --no-cache-dir -r requirements.txt
```

---
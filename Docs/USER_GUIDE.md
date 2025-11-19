# User Guide - AgroPredict

## Getting Started

### Account Creation

1. Open http://127.0.0.1:5000
2. Click "Go to Login"
3. Enter username (minimum 3 characters)
4. Enter password (minimum 6 characters)
5. Click "Register"
6. Use credentials to login

### Important Security Notes

- Keep password confidential
- Use strong, unique passwords
- Do not share login credentials
- Logout when finished

## Using the Application

### Main Workflow

```
Homepage → Login → Upload Image → View Results → Logout
```

### Step 1: Homepage

Features overview and navigation links

### Step 2: Authentication

- Register new account
- Login with existing credentials
- Secure password hashing

### Step 3: Image Upload

1. Click file upload area
2. Select coffee leaf image
3. Accepted formats: PNG, JPG, JPEG, GIF
4. Max file size: 16MB
5. Click "Upload Image"

### Step 4: Processing

- Image preprocessing (224x224 resize)
- AI model inference (1-3 seconds)
- Disease classification
- Confidence calculation

### Step 5: Results Display

Results appear in modal window:

**Components:**
- Uploaded image display
- Disease classification (Healthy/Disease)
- Confidence percentage
- Confidence bar visualization
- Prevention recommendations

### Step 6: Next Actions

- **Upload Another** - Analyze different image
- **Home** - Return to homepage
- **Logout** - Exit account

## Understanding Results

### Disease Classifications

#### Healthy Leaf
- Status: Green indicator
- Meaning: No disease detected
- Action: Continue current practices

#### Coffee Rust
- Status: Red indicator
- Meaning: Fungal infection detected
- Action: Apply fungicide treatments
- Cause: Hemileia vastatrix fungus
- Severity: High priority

#### Red Spider Mite
- Status: Red indicator
- Meaning: Pest infestation
- Action: Apply acaricide treatments
- Cause: Oligonychus yothersi mite
- Severity: High priority

### Confidence Score

- Range: 0-100%
- Higher percentage = More accurate
- 90%+ = High confidence
- 70-90% = Medium confidence
- Below 70% = Low confidence (verify manually)

### Prevention Recommendations

Based on disease classification:
- Fungicide application for rust
- Acaricide application for mites
- Ongoing practices for healthy leaves

## Tips for Best Results

### Image Quality

- Well-lit photographs
- Clear leaf visibility
- Zoom to leaf level
- Avoid shadows
- Single leaf preferred
- Clean lens/camera

### Image Angle

- Top-down angle
- Front-facing view
- Flat positioning
- Fill frame with leaf

### Image Size

- Minimum: 800x600 pixels
- Recommended: 1024x768 or larger
- Maximum: 16MB file size

### Environmental Factors

- Proper lighting
- No motion blur
- Sharp focus on leaf
- Consistent color balance

## Troubleshooting

### Upload Fails

**Issue:** File not accepted

**Solutions:**
- Check file format (PNG, JPG, JPEG, GIF)
- Verify file size under 16MB
- Ensure file not corrupted
- Try different image

### Unexpected Results

**Issue:** Prediction seems incorrect

**Solutions:**
- Verify image quality
- Check leaf is clearly visible
- Ensure proper lighting
- Consult agricultural expert
- Upload different angle

### Processing Slow

**Issue:** Prediction takes longer than expected

**Solutions:**
- Check system resources
- Close other applications
- Verify internet connection
- Restart application
- Try smaller image file

## Account Management

### Changing Login Password

Currently not available. To reset:
1. Contact administrator
2. Provide username for verification
3. Receive temporary password

### Account Deletion

To delete account:
1. Contact administrator
2. Provide username and password
3. Request account deletion
4. Confirm deletion

### Logout

Always logout when finished:
1. Click "Logout" button
2. Or close browser window
3. Session expires after inactivity

---
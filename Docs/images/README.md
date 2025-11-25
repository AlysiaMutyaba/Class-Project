# Screenshots and Images Reference

## Directory Structure

```
docs/
└── images/
    ├── homepage.png
    ├── login.png
    ├── upload.png
    └── prediction.png
```

## Image Placeholders

### homepage.png
**Description:** AgroPredict homepage landing page
**Location:** Should show:
- Application title and branding
- Feature overview
- Navigation links to login
- Project description
- Call-to-action button

**Dimensions:** 1200x800 pixels recommended
**Format:** PNG

---

### login.png
**Description:** Authentication page showing registration and login forms
**Location:** Should display:
- Registration form fields
- Login form fields
- Password input with masking
- Submit buttons
- Background image
- Form validation messages

**Dimensions:** 800x600 pixels recommended
**Format:** PNG

---

### upload.png
**Description:** Image upload interface
**Location:** Should show:
- File upload area
- Drag-and-drop zone
- File type restrictions
- Upload button
- User welcome message
- Navigation menu

**Dimensions:** 900x700 pixels recommended
**Format:** PNG

---

### prediction.png
**Description:** Results modal with disease prediction
**Location:** Should display:
- Analyzed coffee leaf image
- Disease classification (Healthy/Disease)
- Confidence percentage
- Confidence bar visualization
- Prevention recommendations
- Action buttons (Upload Another, Home)
- Modal header and footer

**Dimensions:** 700x900 pixels recommended
**Format:** PNG

---

## Adding Screenshots

### On Windows

1. Use Snipping Tool or ShareX
2. Capture desired screen area
3. Save as PNG to docs/images/
4. Name according to template above

### On macOS

```bash
# Full screenshot
Command + Shift + 3

# Selected area
Command + Shift + 4

# Save to docs/images/
```

### On Linux

```bash
# Using GNOME Screenshot
gnome-screenshot -a

# Using scrot
scrot ~/Class-Project/docs/images/screenshot.png
```

---

## Image Requirements

### Technical Specifications

- **Format:** PNG (lossless compression)
- **Color Mode:** RGB or RGBA
- **Resolution:** 96 DPI minimum
- **Compression:** Optimized for web
- **File Size:** Under 2MB each

### Visual Guidelines

- Show complete UI elements
- Include cursor for interactions
- Avoid sensitive information (passwords, emails)
- Use consistent lighting
- Clear and legible text
- Proper contrast and colors

### File Naming Convention

- Lowercase with underscores
- Descriptive names
- Version suffix if multiple (e.g., `login_v2.png`)
- Include dimensions in filename if needed

---

## Documentation Integration

### Markdown Reference

```markdown
![Screenshot Description](./docs/images/filename.png)
```

### In README

```markdown
## Application Interface

### Homepage

![AgroPredict Homepage](./docs/homepage.png)

### Login Page

![Authentication Page](./docs/login.png)

### Upload Interface

![Image Upload](./docs/upload.png)

### Prediction Results

![Disease Prediction](./docs/Prediction.jpg)
```

---

Document Version: 1.0.0
Last Updated: November 2025
# Copilot / AI Agent Instructions for AgroPredict

This file gives concise, project-specific guidance for AI coding agents (Copilot-style) so you can be productive immediately.

## Quick Project Summary
- Flask web app that serves an image-classification model for coffee leaf disease detection.
- Main server: `app.py` — routes: `/`, `/auth`, `/upload`, `/logout`.
- Model file expected at: `model/my_custom_cnn_windows.pkl` (loaded with `fastai.load_learner`).
- Persistent users stored in SQLite via SQLAlchemy (`User` model in `app.py`).

## Where to look first (key files)
- `app.py` — primary logic: model loading, `is_coffee_leaf` heuristics, upload & prediction flow, `DISEASE_INFO` mapping.
- `README.md` and `Docs/INSTALLATION.md` — setup steps and runtime expectations.
- `templates/` and `static/CSS/` — UI patterns (flash messages, modal results in `upload.html`).

## Important conventions & gotchas (do not change lightly)
- Environment variables: `SECRET_KEY` and `SQLALCHEMY_DATABASE_URI` are read by `app.py`; local dev may rely on defaults (or create `.env`).
- Model loading: `load_learner(model_path, cpu=True)` is used. Ensure `model/my_custom_cnn_windows.pkl` exists and is compatible with the FastAI/PyTorch versions in `requirements.txt`.
- Path hack: `pathlib.PosixPath = pathlib.WindowsPath` is set at the top of `app.py` to avoid Windows path issues when loading the FastAI artifact — be cautious when modifying this line.
- Prediction pipeline: code constructs a PyTorch tensor from a `fastai.PILImage` and calls `learn.model(img_batch)` directly. The agent should preserve this flow or carefully refactor both preprocessing and inference together.
- Labels mapping: `learn.dls.vocab` is used to map prediction indices to labels. The project’s `DISEASE_INFO` keys use underscores (e.g. `Coffee_rust`) — ensure any label normalization matches `learn.dls.vocab` exactly.

## Developer workflows / common commands
- Create virtual env and install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run the app locally: `python app.py` (defaults to `127.0.0.1:5000`).
- Initialize DB (done automatically at startup via `db.create_all()` in `app.py`).
- Test model (README mentions `test_prediction_v2.py`) — note: that file is referenced but not present in the repo. Prefer running the app and uploading a test image or add a short test script that loads `model/my_custom_cnn_windows.pkl`.

## Useful constants & thresholds (used in code)
- Allowed file types: `{'png','jpg','jpeg','gif'}` (see `app.config['ALLOWED_EXTENSIONS']`).
- Max upload size: `16 * 1024 * 1024` (16MB).
- Coffee-leaf validation: `MIN_COFFEE_CONFIDENCE = 0.30` inside the upload flow.
- Model confidence threshold: `CONFIDENCE_THRESHOLD = 0.70` (predictions below this will be rejected via flash).

## When making changes, follow these patterns
- Preserve flash-based UX: `flash(message, category)` + templates consume `get_flashed_messages(with_categories=True)`.
- Use `secure_filename()` for uploads and save under `static/uploads/`.
- If modifying model loading or inference, update both `load_model()` and the inference block in `/upload` to keep preprocessing consistent.
- When adding new disease classes: update `DISEASE_INFO` in `app.py` and ensure model `vocab` order matches keys (or add a normalization mapping layer).

## Integration points & external deps
- FastAI / PyTorch: model artifact compatibility is sensitive to library versions — consult `requirements.txt` when upgrading fastai/torch.
- Flask + SQLAlchemy: sessions are cookie-based and `User` model is the only DB table currently required.

## Missing or fragile elements to flag for maintainers
- `test_prediction_v2.py` is referenced but missing — consider adding a lightweight test harness for offline model checks.
- The `pathlib.PosixPath = pathlib.WindowsPath` override is a non-standard workaround; verify its necessity on CI or Linux hosts.
- `Docs/` includes `INSTALLATION.md` and `USER_GUIDE.md` — prefer canonical instructions there but keep `README.md` in sync.

## How to ask for clarification in PRs
- Point to the exact file & function (e.g. `app.py::is_coffee_leaf`) and include a minimal reproduction (test image or step-by-step commands).

---
If any section is unclear or you'd like the agent to add a `test_prediction.py` harness or adjust model-loading to avoid the `pathlib` override, reply and I'll update this guidance and implement the test script.

# Memory Optimization Guide for Render Free Plan (512MB)

## Changes Made

### 1. **Removed Unused Models & Dependencies**
- ❌ Removed `load_coffee_leaf_detector()` and ResNet50 model (saves ~100MB memory)
- ❌ Removed `torchvision.models` import (not needed)
- ❌ Removed unused packages: `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `requests`, `torchaudio`

### 2. **Memory Management in Code**
- ✅ Added garbage collection (`gc.collect()`) after each prediction
- ✅ Explicitly delete tensors after use to free memory immediately
- ✅ Clear CUDA cache (if GPU accidentally used)

### 3. **Gunicorn Configuration (Dockerfile & render.yaml)**
```bash
gunicorn -w 1 --threads 2 -b 0.0.0.0:5000 \
  --timeout 300 \
  --max-requests 100 \
  --max-requests-jitter 20 \
  --preload \
  app:app
```

**Key settings:**
- `-w 1`: Single worker (reduces memory by ~40-50%)
- `--threads 2`: Handle concurrent requests within single worker
- `--timeout 300`: Allow 5 minutes for slow first request (model loading)
- `--max-requests 100`: Recycle worker after 100 requests to prevent memory leaks
- `--preload`: Load model once before forking (memory efficiency)

### 4. **Expected Memory Usage**
- Base Python + Flask: ~50MB
- PyTorch CPU: ~100-150MB
- FastAI + Model: ~150-200MB
- Working memory: ~50-100MB
- **Total: ~350-500MB** (fits within 512MB limit)

## Deployment Steps

1. **Push changes to GitHub**
   ```bash
   git add .
   git commit -m "Optimize for Render free plan (512MB)"
   git push
   ```

2. **On Render Dashboard:**
   - Go to your service settings
   - Set environment variables:
     - `SECRET_KEY`: (auto-generated or manual)
     - `SQLALCHEMY_DATABASE_URI`: (your Postgres connection string)
   - Deploy the latest commit

3. **Monitor Memory Usage:**
   - Check Render logs for `[CRITICAL] WORKER TIMEOUT` errors
   - If workers timeout, the model is loading (first request takes 30-60s)
   - Subsequent requests should be fast (<2s)

## Additional Tips

### If Still Running Out of Memory:

1. **Use external model storage** (not recommended for free tier):
   - Store model on S3/CDN
   - Download on startup instead of including in Docker image

2. **Reduce model size:**
   - Use model quantization (int8 instead of float32)
   - Use smaller CNN architecture
   - Prune unnecessary layers

3. **Add swap memory** (Render doesn't support this on free tier)

4. **Upgrade to Starter plan** ($7/mo with 512MB → 2GB)

### Monitoring Commands:

Check memory usage in logs:
```bash
# Will show in Render logs if worker gets killed
Worker with pid XXX was killed (signal 9)  # OOM killed
```

## Troubleshooting

### Problem: Worker timeout on first request
**Solution:** This is normal. The 300s timeout should handle it. Wait patiently.

### Problem: Subsequent requests also timing out
**Solution:** Model didn't preload properly. Check logs for `[MODEL ERROR]`.

### Problem: Random crashes after running for hours
**Solution:** Memory leak. The `--max-requests 100` should help by recycling workers.

### Problem: Still getting OOM killed
**Solution:** 
1. Verify `torchaudio`, `pandas`, `scipy` are NOT in installed packages
2. Use `pip list` in Render shell to check
3. Consider removing more FastAI dependencies if desperate

## Performance Expectations

- **First request:** 30-60 seconds (model loading)
- **Subsequent requests:** 1-3 seconds per prediction
- **Concurrent users:** 1-2 simultaneous (with threading)
- **Daily requests:** Unlimited (but slow)

Good luck! The app should now comfortably run on Render's free 512MB plan.

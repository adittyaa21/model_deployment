"""
DEPLOYMENT GUIDE
================

Complete guide for deploying the Student Placement System
"""

# ============================================================================
# DEPLOYMENT GUIDE - Student Placement & Salary Prediction System
# ============================================================================

## 🚀 QUICK DEPLOYMENT SUMMARY

**For Local Testing:**
```bash
streamlit run 03_streamlit_app.py      # Monolithic
# OR
python 04_fastapi_backend.py           # Decoupled backend
streamlit run 05_streamlit_frontend.py # Decoupled frontend
```

**For Cloud Deployment:**
See "Streamlit Cloud Deployment" section below

---

## 📋 DEPLOYMENT OPTIONS

### Option 1: Local Machine
**Best for:** Development, testing, presentations

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (one-time)
python 02_ml_pipeline.py

# 3A. Run monolithic app
streamlit run 03_streamlit_app.py

# 3B. Or run decoupled architecture
# Terminal 1:
python 04_fastapi_backend.py

# Terminal 2:
streamlit run 05_streamlit_frontend.py
```

### Option 2: Streamlit Cloud (Recommended)
**Best for:** Public access, no infrastructure management

**Prerequisites:**
- GitHub account with code pushed to repo
- Streamlit Cloud account (https://streamlit.io/cloud)

**Steps:**

1. **Create GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Set up Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repo and branch
   - Enter file path: `03_streamlit_app.py`
   - Click "Deploy"

3. **Public URL Format:**
   ```
   https://[username]-[repo-name].streamlit.app
   ```

4. **Advanced Configuration** (Optional - `.streamlit/config.toml`)
   ```toml
   [theme]
   primaryColor = "#1f77b4"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f0f2f6"
   textColor = "#2c3e50"
   font = "sans serif"
   
   [client]
   showErrorDetails = false
   toolbarMode = "minimal"
   ```

### Option 3: Docker Deployment
**Best for:** Production, enterprise environments

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create directories
RUN mkdir -p saved_models mlruns

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default: Run monolithic app
CMD ["streamlit", "run", "03_streamlit_app.py"]
```

**Build and Run:**
```bash
# Build image
docker build -t student-placement:latest .

# Run container
docker run -p 8501:8501 -p 8000:8000 student-placement:latest

# Or run specific app
docker run -p 8501:8501 student-placement:latest streamlit run 05_streamlit_frontend.py
```

**Push to Docker Hub:**
```bash
docker tag student-placement:latest username/student-placement:latest
docker push username/student-placement:latest
```

### Option 4: Heroku Deployment (Legacy)
**Note:** Heroku free tier discontinued. Use Railway, Render, or Cloud Run instead.

### Option 5: Google Cloud Run
**Best for:** Serverless, scalable deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/student-placement:latest', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/student-placement:latest']
  
  - name: 'gcr.io/cloud-builders/gke-deploy'
    args:
      - run
      - --filename=k8s/
      - --image=gcr.io/$PROJECT_ID/student-placement:latest
      - --location=us-central1
```

**Deploy:**
```bash
gcloud run deploy student-placement \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🔧 CONFIGURATION FOR DEPLOYMENT

### Environment Variables
Create `.env` file:
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ML Configuration
MODEL_PATH=./saved_models
MLFLOW_TRACKING_URI=http://localhost:5000

# Security
DEBUG_MODE=false
CORS_ORIGINS=*
```

### Streamlit Config (`.streamlit/config.toml`)
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = true

[logger]
level = "info"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

---

## 📊 MONITORING & MAINTENANCE

### Log Monitoring
```bash
# Streamlit logs
tail -f ~/.streamlit/logs/*

# FastAPI logs
# Check console output where server is running
```

### Model Retraining
```bash
# Run pipeline to update models
python 02_ml_pipeline.py

# Stop current app and restart to load new models
```

### MLflow Tracking
```bash
# View experiments and metrics
mlflow ui

# Access at http://localhost:5000
```

---

## 🔒 SECURITY CONSIDERATIONS

### For Production:
1. **API Authentication**
   - Add API key validation
   - Implement JWT tokens
   - Use OAuth2 for social login

2. **Data Protection**
   - Encrypt sensitive inputs
   - Use HTTPS only
   - Implement access controls

3. **Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict/placement")
   @limiter.limit("10/minute")
   async def predict_placement(student: StudentData):
       ...
   ```

4. **Input Validation**
   - Already implemented with Pydantic models
   - Add additional business logic validation

---

## 📈 PERFORMANCE OPTIMIZATION

### Caching
```python
# Streamlit caching
@st.cache_resource
def load_models():
    return load_trained_models()

# FastAPI caching (Redis)
from fastapi_cache2 import FastAPICache
from fastapi_cache2.backends.redis import RedisBackend

# In startup:
FastAPICache.init(RedisBackend(redis_url="redis://localhost"), prefix="fastapi-cache")
```

### Load Balancing
Use Kubernetes or NGINX for multiple container instances:
```nginx
upstream fastapi_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location /api {
        proxy_pass http://fastapi_backend;
    }
}
```

---

## 🧪 PRE-DEPLOYMENT CHECKLIST

- [ ] All dependencies in `requirements.txt`
- [ ] Data files (`A.csv`, `A_targets.csv`) in repo
- [ ] Models trained and saved (`saved_models/`)
- [ ] `.gitignore` configured for sensitive files
- [ ] Tests passing locally
- [ ] README.md complete and accurate
- [ ] API documentation verified (`/docs`)
- [ ] Frontend connects to backend successfully
- [ ] Environment variables configured
- [ ] No hardcoded credentials or sensitive data
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance tested
- [ ] Accessibility checked (WCAG)

---

## 🚨 TROUBLESHOOTING DEPLOYMENT

### Issue: Models not loading
**Solution:**
```bash
python 02_ml_pipeline.py  # Retrain models
# Check saved_models/ directory
```

### Issue: API connection timeout
**Solution:**
```bash
# Verify API is running
curl http://localhost:8000/health

# Check firewall rules
# Verify port 8000 is accessible
```

### Issue: Streamlit Cloud build fails
**Solution:**
```bash
# Check packages in requirements.txt
# Verify Python version >= 3.9
# Remove version conflicts
pip freeze > requirements.txt
```

### Issue: Out of memory
**Solution:**
```python
# Use generators for large files
def load_data():
    yield from large_dataset  # Memory efficient

# Implement data batching in API
@app.post("/predict/batch")
async def batch_predict(file: UploadFile):
    chunk_size = 100  # Process 100 at a time
    for chunk in pd.read_csv(file.file, chunksize=chunk_size):
        # Process chunk
        pass
```

---

## 📞 SUPPORT & RESOURCES

**Official Documentation:**
- Streamlit: https://docs.streamlit.io
- FastAPI: https://fastapi.tiangolo.com
- MLflow: https://mlflow.org/docs
- scikit-learn: https://scikit-learn.org/stable/documentation.html

**Deployment Platforms:**
- Streamlit Cloud: https://streamlit.io/cloud
- Google Cloud Run: https://cloud.google.com/run
- AWS Elastic Container Service: https://aws.amazon.com/ecs/
- Azure Container Instances: https://azure.microsoft.com/en-us/products/container-instances/

---

## ✅ DEPLOYMENT SUCCESS

Your application is successfully deployed when:
- ✅ Frontend loads without errors
- ✅ API endpoints respond correctly
- ✅ Predictions are generated in < 2 seconds
- ✅ Models load automatically on startup
- ✅ Error messages are user-friendly
- ✅ System remains stable under normal load

---

**Last Updated:** April 21, 2024
**Status:** Production Ready

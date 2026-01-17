# Deployment Guide

Deploy the Customer Churn Intelligence Platform for free using cloud services.

---

## Quick Deploy Options

| Component | Service | Cost | Setup Time |
|-----------|---------|------|------------|
| Dashboard | Streamlit Cloud | Free | 5 min |
| API | Render | Free | 10 min |
| Database | Not required | - | - |

---

## Option 1: Streamlit Cloud (Dashboard Only)

**Best for:** Showcasing the dashboard without the API.

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/churn-predictor.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"**

4. **Configure:**
   - Repository: `YOUR_USERNAME/churn-predictor`
   - Branch: `main`
   - Main file path: `dashboard/app.py`

5. **Click "Deploy"**

Your dashboard will be live at: `https://YOUR_APP.streamlit.app`

---

## Option 2: Render (API Backend)

**Best for:** Deploying the FastAPI backend.

### Steps

1. **Go to [render.com](https://render.com)** and sign up

2. **New → Web Service**

3. **Connect your GitHub repo**

4. **Configure:**
   - Name: `churn-predictor-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

5. **Click "Create Web Service"**

Your API will be live at: `https://churn-predictor-api.onrender.com`

---

## Option 3: Railway (Full Stack)

**Best for:** One-click deployment of everything.

### Steps

1. **Go to [railway.app](https://railway.app)**

2. **New Project → Deploy from GitHub**

3. **Select your repository**

4. **Configure two services:**

   **Service 1: API**
   - Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   
   **Service 2: Dashboard**
   - Start Command: `streamlit run dashboard/app.py --server.port $PORT`

5. **Generate domains for each service**

---

## Required Files

Make sure these files exist in your repo:

### `requirements.txt`
Already exists with all dependencies.

### `.streamlit/config.toml`
Created for Streamlit Cloud configuration.

### `Procfile` (for Heroku alternative)
```
web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

---

## Environment Variables

Set these in your deployment platform:

| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.10` | Yes |
| `PORT` | Auto-set | No |

---

## Post-Deployment Checklist

- [ ] Verify API health: `https://YOUR_API/health`
- [ ] Verify API docs: `https://YOUR_API/docs`
- [ ] Test dashboard loads
- [ ] Test prediction endpoint
- [ ] Share your live URL!

---

## Troubleshooting

### "Module not found" error
→ Check `requirements.txt` includes all dependencies

### Dashboard won't load
→ Ensure `dashboard/app.py` is the correct path

### API returns 500
→ Check logs in Render/Railway dashboard

### Model files missing
→ Ensure `models/` folder is committed to git

---

## Free Tier Limits

| Service | Limit |
|---------|-------|
| Streamlit Cloud | Unlimited public apps |
| Render | 750 hours/month, sleeps after 15 min inactivity |
| Railway | $5 free credit/month |

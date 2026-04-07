**Important Note on Architecture:**

This project has a **Python backend** (Flask) and a **static frontend** (HTML/CSS/JS). 

**Vercel Limitations:**
- ✅ Vercel can host the **frontend** (static files)
- ❌ Vercel **cannot** run long-running Python processes (Flask backend)

**Recommended Deployment Strategy:**

1. **Backend**: Deploy to **Railway, Heroku, or Render** (Python-compatible)
2. **Frontend**: Deploy to **Vercel** (static hosting)
   - Update API endpoint in `frontend/app.js` to point to backend URL

**Example:**
```javascript
// frontend/app.js - Update this when deploying
const API = 'https://your-backend-url.railway.app/api';  // Change this
```

**For Vercel Frontend Deployment:**
1. Go to [vercel.com](https://vercel.com)
2. Click "Add New..." → "Project"
3. Select your GitHub repo
4. Set root directory to `frontend/`
5. Deploy

**For Backend Deployment (choose one):**

**Railway:**
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → Import Git Repo
3. Select your repo
4. Railway auto-detects Python
5. Set start command: `cd backend && python app.py`

**Render:**
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repo
4. Build: `pip install -r requirements.txt`
5. Start: `cd backend && python app.py`

**Heroku:**
```bash
heroku create your-app-name
heroku buildpacks:add heroku/python
git push heroku master
```

# VisionEdge

Live object detection over WebRTC. Decode → YOLO inference → draw boxes → stream to browser.

## Run locally

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8000` in a browser (the backend now serves the frontend directly). Click **Start Stream**.

## Deploy live (Render, free tier)

1. Push this repo to GitHub (you've already done this).
2. Go to [render.com](https://render.com), sign up with your GitHub account.
3. **New +** → **Web Service** → select this repo.
4. Render will detect `render.yaml` automatically and pre-fill the build/start commands. If it doesn't, set them manually:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `cd backend && python main.py`
5. Choose the **Free** plan and click **Create Web Service**.
6. Wait for the build to finish (first build installs PyTorch/YOLO, so it can take 5–10 minutes). Once live, Render gives you a URL like `https://visionedge.onrender.com` — that's your working, public demo.

**Free-tier note:** Render's free web services spin down after 15 minutes of inactivity and take ~30–60 seconds to wake back up. That's normal.

## Known limits

- Capped at 5 concurrent viewers to keep the free-tier instance stable.
- Inference runs on CPU, so FPS is modest — that's the documented upgrade path to TensorRT/GPU.

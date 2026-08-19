"""
VisionEdge — deployable backend

Changes from the local-only starter:
  - Serves the frontend (index.html) itself, so ONE deployment covers
    both the API and the page you view it on -- no separate hosting,
    no CORS headaches, no hardcoded localhost URL in the frontend.
  - Defaults to a bundled sample_video.mp4 instead of a webcam, since a
    public server has no webcam of its own.
  - Wraps the frame-read/inference loop in try/except so one bad frame
    doesn't kill the whole stream.
  - Caps concurrent viewers so one deployment doesn't get overwhelmed.
"""

import asyncio
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MAX_CONCURRENT_VIEWERS = 5

app = FastAPI(title="VisionEdge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

relay = MediaRelay()
pcs = set()
model = YOLO("yolov8n.pt")

metrics = {"fps": 0.0, "last_inference_ms": 0.0, "frames_processed": 0, "errors": 0}


def run_inference(frame: np.ndarray) -> np.ndarray:
    start = time.perf_counter()
    try:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}", (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
    except Exception:
        metrics["errors"] += 1
    metrics["last_inference_ms"] = round((time.perf_counter() - start) * 1000, 1)
    metrics["frames_processed"] += 1
    return frame


class AnnotatedVideoTrack(VideoStreamTrack):
    def __init__(self, source):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        self._last_tick = time.perf_counter()

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        try:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = run_inference(frame)
        except Exception:
            metrics["errors"] += 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        now = time.perf_counter()
        metrics["fps"] = round(1.0 / max(now - self._last_tick, 1e-6), 1)
        self._last_tick = now

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class Offer(BaseModel):
    sdp: str
    type: str
    source: str = "sample"


@app.post("/offer")
async def offer(params: Offer):
    if len(pcs) >= MAX_CONCURRENT_VIEWERS:
        raise HTTPException(status_code=503, detail="Viewer limit reached, try again shortly")

    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    if params.source == "sample":
        source = str(BASE_DIR / "sample_video.mp4")
    elif params.source.isdigit():
        source = int(params.source)
    else:
        source = params.source

    track = AnnotatedVideoTrack(source=source)
    pc.addTrack(relay.subscribe(track))

    offer_desc = RTCSessionDescription(sdp=params.sdp.replace("\r\n", "\n").replace("\n", "\r\n"), type=params.type)
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "id": pc_id}


@app.get("/metrics")
async def get_metrics():
    return {**metrics, "active_viewers": len(pcs)}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("shutdown")
async def on_shutdown():
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

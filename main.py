"""
VisionEdge (portfolio-scale starter)
=====================================
Architecture mirrors the target design: decode -> inference -> draw -> WebRTC stream.

WHAT'S REAL vs WHAT'S SUBSTITUTED (be upfront about this on your resume/demo):
  - WebRTC streaming via aiortc: REAL, same library named in the spec.
  - Object detection: REAL YOLO inference (ultralytics), just not compiled to a
    TensorRT engine yet.
  - Video decode: OpenCV/CPU decode, NOT NVDEC hardware decode.
  - Frames live in system RAM and pass through PyTorch on CPU or CUDA if
    available, NOT a CuPy zero-copy VRAM-only pipeline.
  - No DeepStream (that's a full NVIDIA SDK, not something you install
    stand-alone) -- this is a hand-rolled equivalent of the same idea.

UPGRADE PATH (do this once you have access to an NVIDIA GPU + TensorRT):
  1. Export the ultralytics model to ONNX: `model.export(format="onnx")`
  2. Build a TensorRT engine from the ONNX graph with `trtexec`.
  3. Replace `run_inference()` below with a TensorRT execution-context call.
  4. Replace `cv2.VideoCapture` with PyAV + NVIDIA Video Codec SDK (NVDEC) for
     hardware decode, and keep frames as CuPy arrays instead of numpy arrays
     to avoid the CPU<->GPU copy.
  5. Everything else (the aiortc streaming layer, the FastAPI signaling
     endpoint, the frontend) stays the same -- that's the point of building it
     this way first.
"""

import asyncio
import time
import uuid

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="VisionEdge Starter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

relay = MediaRelay()
pcs: set[RTCPeerConnection] = set()

# Load once at startup. yolov8n is the smallest/fastest model -- good for a
# CPU demo. Swap to yolov10 to match the doc once you're on a real GPU box.
model = YOLO("yolov8n.pt")

# Simple in-process metrics so the frontend telemetry panel has real numbers
# instead of hardcoded fakes -- this is the CPU-only stand-in for the
# "Telemetry Dashboard" module in the original spec.
metrics = {"fps": 0.0, "last_inference_ms": 0.0, "frames_processed": 0}


def run_inference(frame: np.ndarray) -> np.ndarray:
    """Runs detection on a single BGR frame and draws boxes in place.

    This is the function you replace with a TensorRT execution call later --
    everything upstream/downstream of this function stays untouched.
    """
    start = time.perf_counter()
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    metrics["last_inference_ms"] = round((time.perf_counter() - start) * 1000, 1)
    metrics["frames_processed"] += 1
    return frame


class AnnotatedVideoTrack(VideoStreamTrack):
    """Reads frames from a source (webcam index or video file path),
    runs inference, and yields annotated frames to aiortc."""

    def __init__(self, source: int | str = 0):
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
                # loop the source video for demo purposes
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()

            frame = run_inference(frame)

            now = time.perf_counter()
            metrics["fps"] = round(1.0 / max(now - self._last_tick, 1e-6), 1)
            self._last_tick = now

        except Exception as e:
            print(f"Error reading frame or running inference: {e}")
            # return a black frame on failure
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


class Offer(BaseModel):
    sdp: str
    type: str
    source: str = "traffic.mp4"  # webcam index as string, or a path to a video file


@app.post("/offer")
async def offer(params: Offer):
    if len(pcs) >= 3:
        raise HTTPException(status_code=503, detail="Maximum concurrent viewers reached. Please try again later.")

    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        if pc.connectionState in ("failed", "closed"):
            pcs.discard(pc)
            await pc.close()

    source: int | str = int(params.source) if params.source.isdigit() else params.source
    track = AnnotatedVideoTrack(source=source)
    pc.addTrack(relay.subscribe(track))

    offer_desc = RTCSessionDescription(sdp=params.sdp, type=params.type)
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "id": pc_id}


@app.get("/metrics")
async def get_metrics():
    return metrics


@app.on_event("shutdown")
async def on_shutdown():
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

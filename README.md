# VisionEdge — Starter (Project 1)

![Demo Preview](preview.png)

**Quick Start:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```
*Then open `index.html` in your browser and click "Start Stream".*

A runnable version of "VisionEdge" from the Axlero project document, scoped to
what you can actually build and demo on a normal machine (no NVIDIA GPU
required to run it).

## Honest scope note
The original spec calls for NVDEC hardware decode, a TensorRT-compiled
inference engine, and a CuPy zero-copy VRAM pipeline — all of which need an
NVIDIA GPU with the CUDA/TensorRT stack installed. This starter keeps the
**same architecture** (decode → inference → draw → WebRTC stream) but uses
CPU-friendly equivalents:

| Spec component            | This starter uses instead     |
|----------------------------|--------------------------------|
| NVDEC hardware decode       | OpenCV `VideoCapture` (CPU)   |
| TensorRT inference engine   | Ultralytics YOLOv8n (PyTorch) |
| CuPy zero-copy pipeline     | Plain numpy arrays            |
| DeepStream                  | Hand-rolled aiortc pipeline   |
| WebRTC streaming (aiortc)   | Same — this part is real      |

`backend/main.py` has inline comments marking exactly which function to
replace when you get access to a CUDA/TensorRT box, so upgrading later is a
drop-in swap, not a rewrite.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Then open `index.html` in a browser and click
**Start Stream**. It defaults to streaming a sample traffic video.

## What to build next (mapped to the original Week 1–2 plan)
1. Swap in a video file of a traffic intersection to simulate the smart-city
   use case instead of a webcam.
2. Add a second and third `AnnotatedVideoTrack` instance to prove multi-stream
   handling (this is the "Week 4: Multi-Stream Orchestration" goal, just done
   with asyncio + CPU inference instead of GPU).
3. Once you have GPU access: export the model to ONNX, build a TensorRT
   engine, and replace `run_inference()`.

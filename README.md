# Autonomous Lab

A browser-based, first-person perception game for testing YOLOv8 driving decisions against photographic city scenes. The game advances the route automatically. The player selects difficulty and a model; **GO/STOP commands come only from model predictions**. Scene labels are used only after inference to grade the run.

## Run

Requires Python 3.12, Node.js 20+, and a machine capable of running PyTorch inference.

```bash
uv sync
npm install
uv run python server.py
```

In a second terminal:

```bash
npm run dev
```

Open `http://localhost:5173`. The included `models/yolov8n.pt` is the official Ultralytics YOLOv8 nano checkpoint. The upload control accepts another trusted Ultralytics `.pt` model (up to 250 MB) and loads it for subsequent runs. Uploaded weights are saved in `models/`. Run `npm run build` for a production frontend build.

## What is simulated

Easy has clear light and crossing scenes. Medium adds animals and an obstacle. Hard adds rain and a misleading green light source. A run evaluates each first-person frame, overlays actual model boxes, applies the policy to detected signals and lane hazards, and automatically advances to the next route checkpoint. A red signal or uncertain signal causes STOP; a person, dog, cat or obstacle in the forward driving path causes STOP. Otherwise the car goes. Green signals do not override a lane hazard. The score is 60% decision correctness and 40% recognition of the scene's required hazards. A final report shows every decision.

The included frames are photorealistic generated road images, not continuous video or a physics-based 3D city. Movement between checkpoints is cinematic rather than a physical driving simulation. The benchmark labels are curated for these five scenes. To expand the benchmark, add a scene image in `public/scenes`, allow its id in `server.py`, and add its expected action and required classes in `src/main.tsx`.

Standard COCO YOLOv8 weights detect people, dogs, cats, cars and traffic lights. They do **not** natively recognize crosswalks, traffic cones, or traffic light colors. The app estimates red/green from pixels inside YOLO traffic-light boxes. A custom model can expose classes such as `crosswalk`, `faixa de pedestres`, `traffic cone`, or `obstacle`; the class mapping lives in `server.py`. Crosswalks are shown as detections but never trigger STOP by themselves. The included baseline may miss small animals and cones or make false detections in the difficult frames; those misses lower its recognition score.

Only load `.pt` files you trust. PyTorch model files can execute code when loaded.

## Implementation

The frontend is React/Vite. The local Python API uses the standard library HTTP server, Pillow, Ultralytics and PyTorch. `/api/status` reports the selected model, `/api/model?name=...` accepts raw `.pt` bytes, and `/api/predict` accepts a whitelisted scene id and returns real boxes, categories, confidence values and the policy decision. No ground-truth field is sent to the inference endpoint.

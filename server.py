from __future__ import annotations

import colorsys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from threading import Lock

from PIL import Image

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / 'models'
SCENE_DIR = ROOT / 'public' / 'scenes'
SCENES = {'clear-green', 'red-pedestrian', 'dog-crossing', 'rainy-decoy', 'cat-cone'}
MAX_MODEL_BYTES = 250 * 1024 * 1024
_model = None
_model_name = None
_model_lock = Lock()

class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def model_files():
    return sorted(path for path in MODEL_DIR.glob('*.pt') if path.is_file())


def load_model(path: Path):
    from ultralytics import YOLO
    return YOLO(str(path))


def current_model():
    global _model, _model_name
    with _model_lock:
        if _model is None:
            files = model_files()
            if not files:
                raise ApiError(409, 'No YOLO model loaded. Add a .pt file to models/ or upload one.')
            try:
                _model = load_model(files[0])
                _model_name = files[0].name
            except Exception as exc:
                raise ApiError(422, f'Could not load {files[0].name}: {exc}') from exc
        return _model


def category_for(label: str):
    name = label.lower().replace('_', ' ').replace('-', ' ').strip()
    if any(term in name for term in ('red light', 'semaforo vermelho', 'traffic light red')):
        return 'red_light'
    if any(term in name for term in ('green light', 'semaforo verde', 'traffic light green')):
        return 'green_light'
    if 'traffic light' in name or 'semaforo' in name:
        return 'traffic_light'
    if name in ('person', 'pedestrian', 'pessoa', 'pedestre'):
        return 'pedestrian'
    if 'crosswalk' in name or 'zebra crossing' in name or 'faixa de pedestres' in name:
        return 'crosswalk'
    if name in ('dog', 'cachorro', 'cao'):
        return 'dog'
    if name in ('cat', 'gato'):
        return 'cat'
    if any(term in name for term in ('obstacle', 'traffic cone', 'cone', 'barrier', 'debris', 'roadblock', 'obstaculo')):
        return 'obstacle'
    return 'other'


def light_color(image: Image.Image, box):
    width, height = image.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 'unknown'
    crop = image.crop((x1, y1, x2, y2)).convert('RGB')
    crop.thumbnail((60, 100))
    red = green = 0
    for r, g, b in crop.getdata():
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if s < 0.42 or v < 0.35:
            continue
        if h < 0.055 or h > 0.95:
            red += 1
        elif 0.30 < h < 0.54:
            green += 1
    if max(red, green) < 2 or max(red, green) < min(red, green) * 1.35:
        return 'unknown'
    return 'red' if red > green else 'green'


def decide(detections):
    reasons = []
    has_colored_light = any(det['category'] in ('red_light', 'green_light') for det in detections)
    for det in detections:
        category = det['category']
        x1, y1, x2, y2 = det['box']
        center_x, bottom_y = (x1 + x2) / 2, y2
        in_path = 0.22 < center_x < 0.78 and bottom_y > 0.28
        if category == 'red_light':
            reasons.append('Red traffic light detected')
        elif category == 'traffic_light' and not has_colored_light:
            reasons.append('Traffic light state uncertain')
        elif category in ('pedestrian', 'dog', 'cat', 'obstacle') and in_path:
            reasons.append(f'{category.replace("_", " ").title()} in the driving path')
    return {'action': 'STOP' if reasons else 'GO', 'reasons': reasons or ['No stop hazard predicted in the driving path']}


def status():
    return {'ready': bool(_model or model_files()), 'model': _model_name or (model_files()[0].name if model_files() else None)}


def upload_model(filename: str, data: bytes):
    global _model, _model_name
    if Path(filename).suffix.lower() != '.pt':
        raise ApiError(400, 'Upload an Ultralytics YOLO .pt model.')
    if len(data) > MAX_MODEL_BYTES:
        raise ApiError(413, 'Model must be 250 MB or smaller.')
    MODEL_DIR.mkdir(exist_ok=True)
    safe_name = Path(filename).name
    target = MODEL_DIR / f'uploaded-{safe_name}'
    temporary = MODEL_DIR / f'.{target.stem}.partial.pt'
    temporary.write_bytes(data)
    try:
        candidate = load_model(temporary)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ApiError(422, f'Invalid YOLO model: {exc}') from exc
    temporary.replace(target)
    with _model_lock:
        _model, _model_name = candidate, target.name
    return {'ready': True, 'model': target.name}


def predict(scene_id: str, confidence: float = 0.25):
    if scene_id not in SCENES:
        raise ApiError(404, 'Unknown scene')
    model = current_model()
    image = Image.open(SCENE_DIR / f'{scene_id}.jpg').convert('RGB')
    try:
        with _model_lock:
            result = model.predict(image, conf=max(0.05, min(confidence, 0.9)), verbose=False)[0]
    except Exception as exc:
        raise ApiError(500, f'Inference failed: {exc}') from exc
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = str(result.names[class_id])
        category = category_for(label)
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        if category == 'traffic_light':
            color = light_color(image, xyxy)
            if color != 'unknown':
                category = f'{color}_light'
        detections.append({'label': label, 'category': category, 'confidence': round(float(box.conf[0]), 3), 'box': [round(xyxy[0] / image.width, 4), round(xyxy[1] / image.height, 4), round(xyxy[2] / image.width, 4), round(xyxy[3] / image.height, 4)]})
    detections.sort(key=lambda item: item['confidence'], reverse=True)
    decision = decide(detections)
    return {'scene_id': scene_id, 'model': _model_name, 'detections': detections, **decision}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode()
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            if urlparse(self.path).path == '/api/status':
                self.send_json(200, status())
            else:
                self.send_json(404, {'detail': 'Unknown endpoint'})
        except ApiError as exc:
            self.send_json(exc.status, {'detail': str(exc)})

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > MAX_MODEL_BYTES + 1024:
                raise ApiError(413, 'Request is too large')
            data = self.rfile.read(length)
            url = urlparse(self.path)
            if url.path == '/api/model':
                filename = parse_qs(url.query).get('name', ['uploaded.pt'])[0]
                self.send_json(200, upload_model(filename, data))
            elif url.path == '/api/predict':
                request = json.loads(data)
                self.send_json(200, predict(request['scene_id'], float(request.get('confidence', 0.25))))
            else:
                self.send_json(404, {'detail': 'Unknown endpoint'})
        except ApiError as exc:
            self.send_json(exc.status, {'detail': str(exc)})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(400, {'detail': str(exc)})


if __name__ == '__main__':
    print('Inference API listening on http://127.0.0.1:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()

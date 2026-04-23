import os
import cv2
import json
import uuid
import time
import base64
try:
    import serial
except ImportError:
    serial = None  # pyserial not installed; GPS will rely on browser fallback
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor
# Change this:
from datetime import datetime
# To this:
from datetime import datetime, timezone
from collections import deque  # ← required for _live_history rolling context

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER']  = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'],  exist_ok=True)

# ─────────────────────────── Configuration ────────────────────────────────────
ROBOFLOW_API_KEY   = os.getenv("ROBOFLOW_API_KEY",   "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY",       "")
COLAB_MODEL_1_URL  = os.getenv("COLAB_MODEL_1_URL",  "")   # xView2
COLAB_MODEL_2_URL  = os.getenv("COLAB_MODEL_2_URL",  "")   # YOLOv8 Fire & Smoke
COLAB_MODEL_3_URL  = os.getenv("COLAB_MODEL_3_URL",  "")   # LADI-v2

# GPS Serial config for NEO-6M on Raspberry Pi
GPS_SERIAL_PORT = os.getenv("GPS_SERIAL_PORT", "/dev/serial0")
GPS_BAUD_RATE   = int(os.getenv("GPS_BAUD_RATE", "9600"))

# ── Local YOLOv8 Fire & Smoke model (optional — loads when weight file exists) ─
# On Pi this will typically be absent; on a dev machine or Pi 5 it can be used
# as a full offline fallback when COLAB_MODEL_2_URL is unreachable.
MODEL_2_PATH = os.getenv(
    "MODEL_2_PATH",
    r"c:\Abhi\Robo_Xaviers (1)-20260423T124018Z-3-001\Robo_Xaviers (1)\Model\1\YOLOv8-Fire-and-Smoke-Detection\runs\detect\train\weights\best.pt"
)
model2 = None
if os.path.exists(MODEL_2_PATH):
    try:
        from ultralytics import YOLO
        model2 = YOLO(MODEL_2_PATH)
        print(f"✅ Local Model 2 (Fire & Smoke) loaded from {MODEL_2_PATH}")
    except Exception as _e:
        print(f"⚠️  Could not load local Model 2: {_e}")
else:
    print(f"⚠️  Local Model 2 not found at {MODEL_2_PATH} — will use Colab remote only")

# ─────────────────────────── GPS State ────────────────────────────────────────
# "source" field: "hardware" | "browser" | "none"
gps_data = {
    "lat": None, "lon": None, "fix": False,
    "satellites": 0, "source": "none", "timestamp": None
}
gps_lock = threading.Lock()

# ─────────────────────────── Zone / Dispatch State ────────────────────────────
zones_db   = {}
zones_lock = threading.Lock()

# ─────────────────────── GPS NEO-6M NMEA Parser ───────────────────────────────
def parse_nmea_gga(sentence):
    """Parse $GPGGA / $GNGGA NMEA sentence from a NEO-6M GPS module."""
    try:
        parts = sentence.split(',')
        if parts[0] not in ('$GPGGA', '$GNGGA'):
            return None
        if not parts[2] or not parts[4]:
            return None

        raw_lat = float(parts[2])
        lat_deg = int(raw_lat / 100)
        lat = lat_deg + (raw_lat - lat_deg * 100) / 60.0
        if parts[3] == 'S':
            lat = -lat

        raw_lon = float(parts[4])
        lon_deg = int(raw_lon / 100)
        lon = lon_deg + (raw_lon - lon_deg * 100) / 60.0
        if parts[5] == 'W':
            lon = -lon

        sats        = int(parts[7]) if parts[7] else 0
        fix_quality = int(parts[6]) if parts[6] else 0

        return {"lat": round(lat, 6), "lon": round(lon, 6),
                "fix": fix_quality > 0, "satellites": sats}
    except Exception:
        return None


def gps_reader_thread():
    """
    Background daemon: continuously reads NMEA data from NEO-6M via UART.
    If the port cannot be opened initially, it will keep retrying every 5 seconds.
    """
    global gps_data
    if serial is None:
        print("⚠️  pyserial not installed — GPS will rely on browser geolocation fallback.")
        return

    while True:
        try:
            ser = serial.Serial(GPS_SERIAL_PORT, GPS_BAUD_RATE, timeout=1)
            print(f"✅ GPS NEO-6M connected on {GPS_SERIAL_PORT} @ {GPS_BAUD_RATE} baud")
            
            while True:
                try:
                    line = ser.readline().decode('ascii', errors='replace').strip()
                    if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                        parsed = parse_nmea_gga(line)
                        if parsed and parsed['fix']:
                            with gps_lock:
                                gps_data = {
                                    **parsed,
                                    "source":    "hardware",
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                }
                except serial.SerialException:
                    print("⚠️  GPS serial connection lost. Reconnecting...")
                    break  # Break inner loop to trigger reconnect
                except Exception:
                    pass
                time.sleep(0.1)
                
        except Exception as e:
            print(f"⚠️  GPS serial open failed: {e} — retrying in 5 seconds...")
            time.sleep(5)


# Start GPS reader daemon at import time
_gps_thread = threading.Thread(target=gps_reader_thread, daemon=True, name="GPSReader")
_gps_thread.start()

# ─────────────────────────── Helper Functions ─────────────────────────────────
def call_roboflow(image, model_id, confidence=40):
    if not ROBOFLOW_API_KEY:
        return []
    try:
        max_dim = 640  # Smaller resolution for Pi performance
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_base64 = base64.b64encode(buffer)
        url = (f"https://detect.roboflow.com/{model_id}"
               f"?api_key={ROBOFLOW_API_KEY}&confidence={confidence}")
        res = requests.post(url, data=img_base64,
                            headers={'Content-Type': 'application/x-www-form-urlencoded'},
                            timeout=10)
        res.raise_for_status()
        return res.json().get('predictions', [])
    except Exception as e:
        print(f"Roboflow Error ({model_id}): {e}")
        return []


def call_groq_vision(image):
    if not GROQ_API_KEY:
        return "Groq API key missing."
    try:
        max_dim = 512
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        url     = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": ("Analyze this disaster scene. Identify: building damage, road damage, "
                          "fire, smoke, flooding, debris. Rate severity 1-10 for each detected "
                          "element. Be concise.")},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}],
            "max_tokens": 150, "temperature": 0.2
        }
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code != 200:
            print(f"Groq Vision Error Details: {res.text}")
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Groq Vision Error: {e}")
        return "Vision verification failed."


def call_groq_summary(results, history=None):
    """
    Generate a 2-3 sentence scene summary.
    Optionally accepts a rolling `history` list of recent summaries so the
    model can maintain situational context across live-feed frames.
    """
    if not GROQ_API_KEY:
        return "Groq API key missing."
    try:
        url     = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        sys_content = ("You are a disaster intelligence AI. Summarize the following raw "
                       "detections and vision verification in 2-3 sentences to tell the user "
                       "exactly what is happening in the scene.")
        if history:
            sys_content += ("\n\nContext of recent events:\n" +
                            "\n".join([f"- {h}" for h in history]))

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": json.dumps(results)}
            ],
            "max_tokens": 150, "temperature": 0.2
        }
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Groq Summary Error: {e}")
        return "Summary generation failed."


def call_colab_model(file_path, url, endpoint):
    if not url:
        return None
    url = url.rstrip('/') + endpoint
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'image/jpeg')}
            res = requests.post(url, files=files, timeout=15)
            res.raise_for_status()
            return res.json()
    except Exception as e:
        print(f"Colab Model Error ({url}): {e}")
        return None


def process_frame(frame, file_path, selected_models, colab_urls, history=None):
    """
    Run all selected models in parallel on a single frame.
    `history` is an optional list of recent summary strings used by
    call_groq_summary to maintain rolling situational awareness.
    """
    results = {"detections": [], "classifications": []}

    def run_m1():
        if "model1" in selected_models:
            url  = colab_urls.get("model1") or COLAB_MODEL_1_URL
            data = call_colab_model(file_path, url, "/assess")
            if data and 'detections' in data:
                for d in data['detections']:
                    results["detections"].append({
                        "source":     "Model 1 (xView2)",
                        "class":      d.get('class'),
                        "confidence": d.get('confidence'),
                        "box":        d.get('box')
                    })

    def run_m2():
        """
        Fire & Smoke detection.
        Priority order:
          1. Remote Colab YOLOv8 server (if URL configured)
          2. Local YOLOv8 weights (if model2 loaded — e.g. Pi 5 / dev machine)
        """
        if "model2" not in selected_models:
            return
        url = colab_urls.get("model2") or COLAB_MODEL_2_URL
        if url:
            data = call_colab_model(file_path, url, "/detect")
            if data and 'detections' in data:
                for d in data['detections']:
                    results["detections"].append({
                        "source":     "Model 2 (Fire/Smoke)",
                        "class":      d.get('class'),
                        "confidence": d.get('confidence'),
                        "box":        d.get('box')
                    })
                return  # Served by Colab — no need for local fallback
        # Local fallback
        if model2:
            preds = model2.predict(source=frame, conf=0.4, verbose=False)
            for r in preds:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf   = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    results["detections"].append({
                        "source":     "Model 2 (Fire/Smoke)",
                        "class":      model2.names[cls_id],
                        "confidence": round(conf, 3),
                        "box":        {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                    })

    def run_m3():
        if "model3" in selected_models:
            url  = colab_urls.get("model3") or COLAB_MODEL_3_URL
            data = call_colab_model(file_path, url, "/predict")
            if data:
                if 'all_scores' in data and len(data['all_scores']) > 0:
                    filtered = [d for d in data['all_scores']
                                if not any(w in d.get('class', '').lower()
                                           for w in ['water', 'flood'])]
                    for d in filtered[:3]:
                        results["classifications"].append({
                            "source":     "Model 3 (LADI-v2)",
                            "class":      d.get('class'),
                            "confidence": d.get('confidence')
                        })
                elif 'top_predictions' in data:
                    for d in data['top_predictions']:
                        if any(w in d.get('class', '').lower() for w in ['water', 'flood']):
                            continue
                        results["classifications"].append({
                            "source":     "Model 3 (LADI-v2)",
                            "class":      d.get('class'),
                            "confidence": d.get('confidence')
                        })

    def run_m4():
        if "model4" in selected_models:
            preds = call_roboflow(frame, "ambulance-4bova/1")
            for p in preds:
                w, h = p['width'], p['height']
                results["detections"].append({
                    "source":     "Model 4 (Ambulance)",
                    "class":      p['class'],
                    "confidence": p['confidence'],
                    "box": {
                        "xmin": int(p['x'] - w / 2), "ymin": int(p['y'] - h / 2),
                        "xmax": int(p['x'] + w / 2), "ymax": int(p['y'] + h / 2)
                    }
                })

    def run_m5():
        if "model5" in selected_models:
            preds = call_roboflow(frame, "vest-5byyt/1")
            for p in preds:
                w, h = p['width'], p['height']
                results["detections"].append({
                    "source":     "Model 5 (Vest)",
                    "class":      p['class'],
                    "confidence": p['confidence'],
                    "box": {
                        "xmin": int(p['x'] - w / 2), "ymin": int(p['y'] - h / 2),
                        "xmax": int(p['x'] + w / 2), "ymax": int(p['y'] + h / 2)
                    }
                })

    def run_m6():
        if "model6" in selected_models:
            vision_text = call_groq_vision(frame)
            results["vision_verification"] = vision_text

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.submit(run_m1)
        executor.submit(run_m2)
        executor.submit(run_m3)
        executor.submit(run_m4)
        executor.submit(run_m5)
        executor.submit(run_m6)

    if "model6" in selected_models:
        summary = call_groq_summary(results, history)  # ← history passed through
        results["summary"] = summary

    return results


def draw_boxes(img, detections):
    colors = {
        "Model 1 (xView2)":     (255,   0,   0),
        "Model 2 (Fire/Smoke)": (  0,   0, 255),   # ← was missing on Pi
        "Model 4 (Ambulance)":  (  0, 255, 255),
        "Model 5 (Vest)":       (255,   0, 255),
    }
    annotated = img.copy()
    for d in detections:
        box = d.get('box')
        if not box:
            continue
        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        color = colors.get(d['source'], (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} ({d['confidence']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


# ─────────────────────────── Zone Scoring Engine ──────────────────────────────
DAMAGE_WEIGHTS = {
    "building_damage": 0.35,
    "road_damage":     0.25,
    "fire":            0.20,
    "smoke":           0.05,
    "flood":           0.10,
    "debris":          0.05,
}

DAMAGE_KEYWORDS = {
    "building_damage": ["building", "structure", "collapse", "damaged_building",
                        "destroyed", "major-damage", "minor-damage", "un-classified"],
    "road_damage":     ["road", "crack", "pothole", "road_damage", "infrastructure"],
    "fire":            ["fire", "flame", "blaze"],
    "smoke":           ["smoke", "haze"],
    "flood":           ["flood", "water", "inundation"],
    "debris":          ["debris", "rubble", "wreckage"],
}


def classify_detection(det_class):
    det_lower = det_class.lower() if det_class else ""
    for category, keywords in DAMAGE_KEYWORDS.items():
        for kw in keywords:
            if kw in det_lower:
                return category
    return None


def compute_zone_score(detections, classifications):
    category_scores = {cat: 0.0 for cat in DAMAGE_WEIGHTS}
    category_counts = {cat: 0   for cat in DAMAGE_WEIGHTS}

    for det in detections:
        cat = classify_detection(det.get('class', ''))
        if cat:
            category_scores[cat] += float(det.get('confidence', 0))
            category_counts[cat] += 1

    for cls in classifications:
        cat = classify_detection(cls.get('class', ''))
        if cat:
            category_scores[cat] += float(cls.get('confidence', 0))
            category_counts[cat] += 1

    for cat in category_scores:
        if category_counts[cat] > 0:
            category_scores[cat] /= category_counts[cat]

    total_score = sum(category_scores[cat] * DAMAGE_WEIGHTS[cat] for cat in DAMAGE_WEIGHTS)

    has_building = category_counts["building_damage"] > 0
    has_road     = category_counts["road_damage"]     > 0

    if has_building or (has_road and total_score > 0.4):
        color = "red"
    elif has_road:
        color = "yellow"
    else:
        color = "green"

    return {
        "total_score":     round(total_score, 4),
        "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        "category_counts": category_counts,
        "color":           color,
    }


def suggest_action(color, score, category_counts):
    if color == "red":
        if category_counts.get("fire", 0) > 0:
            return "Deploy fire response unit. Evacuate nearby zones immediately."
        elif (category_counts.get("building_damage", 0) > 0 and
              category_counts.get("road_damage",     0) > 0):
            return ("Send medical supplies + structural assessment team. "
                    "Road impassable, reroute via alternate.")
        elif category_counts.get("building_damage", 0) > 0:
            return "Send medical supplies + search-and-rescue team."
        else:
            return "High severity — deploy emergency response team."
    elif color == "yellow":
        if category_counts.get("road_damage", 0) > 0:
            return "Road impassable. Deploy road-clearing crew. Reroute traffic."
        return "Moderate damage — monitor and dispatch road repair crew."
    else:
        return "Area appears clear. Continue aerial surveillance."


# =============================================================================
# LIVE FEED — "Latest-Frame Wins" Architecture
# =============================================================================
# Inference takes ~10 s.  Processing frames sequentially causes results to be
# perpetually stale and builds an unbounded queue.
#
# Solution: single-slot frame buffer.
#   • Browser POSTs frames every 2 s  → stored in _live_slot (overwrites old)
#   • One background worker thread picks the LATEST slot and runs inference
#   • If 5 new frames arrive during a 10 s inference, only frame #6 is next
#   • Frontend polls /live_status every 2 s — never blocks
# =============================================================================

_live_slot_lock   = threading.Lock()
_live_slot        = None        # {"frame": ndarray, "meta": {...}}

_live_result_lock = threading.Lock()
_live_result      = {
    "status":           "idle",
    "frames_received":  0,
    "frames_processed": 0,
    "frames_dropped":   0,
    "worker_busy":      False,
}

_live_event   = threading.Event()   # signals worker that a new frame arrived
_live_history = deque(maxlen=3)     # rolling context: last 3 summaries


def _live_worker():
    """
    Daemon thread — runs forever.
    Waits for a frame signal, grabs the LATEST slot (discards all queued-up
    older ones), runs full inference, stores result, loops.
    """
    print("🎥 Live feed worker started (latest-frame-wins mode)")
    while True:
        _live_event.wait()
        _live_event.clear()

        with _live_slot_lock:
            slot = _live_slot       # grab reference to the newest slot
        if slot is None:
            continue

        frame = slot["frame"]
        meta  = slot["meta"]

        with _live_result_lock:
            _live_result["worker_busy"] = True

        try:
            temp_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                f"lw_{uuid.uuid4().hex[:8]}.jpg"
            )
            cv2.imwrite(temp_path, frame)

            try:
                results = process_frame(
                    frame, temp_path,
                    meta["models"], meta["colab_urls"],
                    history=list(_live_history)   # pass rolling context
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Update rolling history with the new summary
            if results.get("summary"):
                _live_history.append(results["summary"])

            annotated = draw_boxes(frame, results['detections'])
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            annotated_b64 = base64.b64encode(buf).decode('utf-8')

            zone_scoring = compute_zone_score(
                results['detections'], results['classifications']
            )
            zone_id = f"LIVE-{uuid.uuid4().hex[:6].upper()}"
            action  = suggest_action(
                zone_scoring['color'],
                zone_scoring['total_score'],
                zone_scoring['category_counts']
            )

            zone_entry = {
                "zone_id":              zone_id,
                "lat":                  meta["coords"]["lat"],
                "lon":                  meta["coords"]["lon"],
                "score":                zone_scoring["total_score"],
                "color":                zone_scoring["color"],
                "category_scores":      zone_scoring["category_scores"],
                "category_counts":      zone_scoring["category_counts"],
                "action":               action,
                "override":             None,
                "timestamp":            datetime.now(timezone.utc).isoformat(),
                "image_url":            None,
                "detections_count":     len(results['detections']),
                "classifications_count":len(results['classifications']),
            }

            if (len(results['detections']) > 0 or
                    len(results['classifications']) > 0):
                with zones_lock:
                    zones_db[zone_id] = zone_entry

            with _live_result_lock:
                _live_result.update({
                    "status":             "ok",
                    "zone":               zone_entry,
                    "detections":         results['detections'],
                    "classifications":    results['classifications'],
                    "summary":            results.get('summary', ''),
                    "vision_verification":results.get('vision_verification', ''),
                    "annotated_frame":    f"data:image/jpeg;base64,{annotated_b64}",
                    "frames_processed":   _live_result["frames_processed"] + 1,
                    "worker_busy":        False,
                })

        except Exception as e:
            print(f"Live worker error: {e}")
            with _live_result_lock:
                _live_result.update({
                    "status":      "error",
                    "error":       str(e),
                    "worker_busy": False,
                })


# Start live-feed worker daemon once at import time
_worker_thread = threading.Thread(
    target=_live_worker, daemon=True, name="LiveWorker"
)
_worker_thread.start()


# ─────────────────────────────── Routes ───────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── GPS ──────────────────────────────────────────────────────────────────────

@app.route('/gps', methods=['GET'])
def get_gps():
    """Return current GPS coordinates (hardware NEO-6M or browser fallback)."""
    with gps_lock:
        return jsonify(gps_data)


@app.route('/gps/browser', methods=['POST'])
def set_browser_gps():
    """
    Accept coordinates pushed from the browser's Geolocation API.

    Policy (Pi-specific):
      • If we already have a live hardware GPS fix, ignore the browser update
        so the more accurate NEO-6M data is preserved.
      • If no hardware fix is present (port unavailable or no satellite lock),
        accept the browser coordinates so the system is never location-blind.
    """
    global gps_data
    data = request.get_json()
    lat  = data.get('lat')
    lon  = data.get('lon')
    if lat is None or lon is None:
        return jsonify({"error": "Missing lat/lon"}), 400

    with gps_lock:
        # Respect hardware GPS when it has a valid fix
        if gps_data.get('fix') and gps_data.get('source') == 'hardware':
            return jsonify({"status": "ok", "note": "hardware GPS active — browser coordinates ignored"})
        gps_data = {
            "lat":       round(float(lat), 6),
            "lon":       round(float(lon), 6),
            "fix":       True,
            "satellites":0,
            "source":    "browser",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    return jsonify({"status": "ok"})


# ── File Upload / Inference ───────────────────────────────────────────────────

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    selected_models = request.form.get('models', 'all')
    if selected_models == 'all':
        selected_models = ['model1', 'model2', 'model3', 'model4', 'model5', 'model6']
    else:
        selected_models = json.loads(selected_models)

    colab_urls = {
        "model1": request.form.get('url_model1', ''),
        "model2": request.form.get('url_model2', ''),
        "model3": request.form.get('url_model3', ''),
    }

    # GPS: Prefer hardware GPS, then manual form values, then browser fallback
    with gps_lock:
        hw_lat, hw_lon = gps_data.get("lat"), gps_data.get("lon")
        hw_fix = gps_data.get("fix") and gps_data.get("source") == "hardware"

    manual_lat = request.form.get('latitude',  '')
    manual_lon = request.form.get('longitude', '')

    if hw_fix and hw_lat is not None and hw_lon is not None:
        coords = {"lat": hw_lat, "lon": hw_lon}
    elif manual_lat and manual_lon:
        coords = {"lat": float(manual_lat), "lon": float(manual_lon)}
    else:
        with gps_lock:
            coords = {"lat": gps_data["lat"], "lon": gps_data["lon"]}

    ext      = file.filename.rsplit('.', 1)[-1].lower()
    filename = f"{uuid.uuid4().hex[:8]}.{ext}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # ── IMAGE ────────────────────────────────────────────────────────────────
    if ext in ['jpg', 'jpeg', 'png']:
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        results       = process_frame(img, upload_path, selected_models, colab_urls)
        annotated_img = draw_boxes(img, results['detections'])

        y_offset = 30
        for cls in results['classifications']:
            text = f"{cls['class']} ({cls['confidence']:.2f})"
            cv2.putText(annotated_img, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        result_name = f"result_{filename}"
        cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], result_name), annotated_img)

        zone_scoring = compute_zone_score(results['detections'], results['classifications'])
        zone_id      = f"ZONE-{uuid.uuid4().hex[:6].upper()}"
        action       = suggest_action(
            zone_scoring['color'], zone_scoring['total_score'],
            zone_scoring['category_counts']
        )

        zone_entry = {
            "zone_id":               zone_id,
            "lat":                   coords["lat"],
            "lon":                   coords["lon"],
            "score":                 zone_scoring["total_score"],
            "color":                 zone_scoring["color"],
            "category_scores":       zone_scoring["category_scores"],
            "category_counts":       zone_scoring["category_counts"],
            "action":                action,
            "override":              None,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "image_url":             f"/static/results/{result_name}",
            "detections_count":      len(results['detections']),
            "classifications_count": len(results['classifications']),
        }

        with zones_lock:
            zones_db[zone_id] = zone_entry

        results["zone"]        = zone_entry
        results["coordinates"] = coords

        return jsonify({
            "status":      "success",
            "type":        "image",
            "image_url":   f"/static/results/{result_name}",
            "json_output": results
        })

    # ── VIDEO ────────────────────────────────────────────────────────────────
    elif ext in ['mp4', 'avi', 'mov', 'webm', 'mkv']:
        cap    = cv2.VideoCapture(upload_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result_name = f"result_{filename}"
        out_path    = os.path.join(app.config['RESULT_FOLDER'], result_name)
        fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
        out         = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        all_results    = []
        frame_count    = 0
        frame_skip     = max(int(fps), 1)   # 1 sampled frame per second
        last_annotated = None
        max_frames     = int(fps * 15)      # Pi cap: 15 s (main uses 10 s)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                temp_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    f"vf_{uuid.uuid4().hex[:6]}.jpg"
                )
                cv2.imwrite(temp_path, frame)

                results        = process_frame(frame, temp_path, selected_models, colab_urls)
                last_annotated = draw_boxes(frame, results['detections'])

                zone_scoring = compute_zone_score(
                    results['detections'], results['classifications']
                )
                zone_id = f"ZONE-{uuid.uuid4().hex[:6].upper()}"
                action  = suggest_action(
                    zone_scoring['color'], zone_scoring['total_score'],
                    zone_scoring['category_counts']
                )

                zone_entry = {
                    "zone_id":               zone_id,
                    "lat":                   coords["lat"],
                    "lon":                   coords["lon"],
                    "score":                 zone_scoring["total_score"],
                    "color":                 zone_scoring["color"],
                    "category_scores":       zone_scoring["category_scores"],
                    "category_counts":       zone_scoring["category_counts"],
                    "action":                action,
                    "override":              None,
                    "timestamp":             datetime.now(timezone.utc).isoformat(),
                    "image_url":             None,
                    "detections_count":      len(results['detections']),
                    "classifications_count": len(results['classifications']),
                }

                if (len(results['detections']) > 0 or
                        len(results['classifications']) > 0):
                    with zones_lock:
                        zones_db[zone_id] = zone_entry

                all_results.append({
                    "time_sec": round(frame_count / fps, 2),
                    "results":  results,
                    "zone":     zone_entry
                })

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            out.write(last_annotated if last_annotated is not None else frame)
            frame_count += 1
            if frame_count >= max_frames:
                break

        cap.release()
        out.release()

        final_mp4  = f"web_{result_name}"
        final_path = os.path.join(app.config['RESULT_FOLDER'], final_mp4)
        ret_code   = os.system(
            f"ffmpeg -y -i {out_path} -vcodec libx264 -an {final_path} 2>/dev/null"
        )
        video_url = (f"/static/results/{final_mp4}"
                     if ret_code == 0 and os.path.exists(final_path)
                     else f"/static/results/{result_name}")

        return jsonify({
            "status":      "success",
            "type":        "video",
            "video_url":   video_url,
            "json_output": {"video_events": all_results, "coordinates": coords}
        })

    else:
        return jsonify({"error": "Unsupported file format. Use jpg/png/mp4/avi/mov."}), 400


# ── Live Feed ─────────────────────────────────────────────────────────────────

@app.route('/live_push', methods=['POST'])
def live_push():
    """
    Accepts a webcam frame from the browser (base64 JPEG in JSON body).
    Returns IMMEDIATELY — inference happens asynchronously in _live_worker.
    Call this every 2 seconds from the browser regardless of inference speed.
    """
    global _live_slot
    data = request.get_json(silent=True)
    if not data or 'frame' not in data:
        return jsonify({"error": "No frame"}), 400

    try:
        fb64 = data['frame']
        if ',' in fb64:
            fb64 = fb64.split(',', 1)[1]
        img_bytes = base64.b64decode(fb64)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Bad frame"}), 400

        # Prefer hardware NEO-6M GPS if available and has fix
        with gps_lock:
            hw_lat, hw_lon = gps_data.get("lat"), gps_data.get("lon")
            hw_fix = gps_data.get("fix") and gps_data.get("source") == "hardware"

        lat = data.get('lat')
        lon = data.get('lon')

        if hw_fix and hw_lat is not None and hw_lon is not None:
            coords = {"lat": hw_lat, "lon": hw_lon}
        elif lat and lon:
            coords = {"lat": float(lat), "lon": float(lon)}
        else:
            with gps_lock:
                coords = {"lat": gps_data["lat"], "lon": gps_data["lon"]}

        new_slot = {
            "frame": frame,
            "meta": {
                "coords":     coords,
                "models":     data.get('models', ['model1', 'model2', 'model3',
                                                  'model4', 'model5', 'model6']),
                "colab_urls": {
                    "model1": data.get('url_model1', ''),
                    "model2": data.get('url_model2', ''),
                    "model3": data.get('url_model3', ''),
                }
            }
        }

        with _live_slot_lock:
            old       = _live_slot
            _live_slot = new_slot
            dropped   = old is not None     # a frame was silently overwritten

        with _live_result_lock:
            _live_result["frames_received"] += 1
            if dropped:
                _live_result["frames_dropped"] += 1

        _live_event.set()   # wake worker

        return jsonify({
            "status":         "queued",
            "worker_busy":    _live_result.get("worker_busy",    False),
            "frames_dropped": _live_result.get("frames_dropped", 0),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/live_status', methods=['GET'])
def live_status():
    """Poll every 2 s from the browser to get the latest inference result."""
    with _live_result_lock:
        return jsonify(dict(_live_result))


@app.route('/live_frame', methods=['POST'])
def live_frame_compat():
    """Backwards-compatibility alias for /live_push — keeps old clients working."""
    return live_push()


# ── Zones ─────────────────────────────────────────────────────────────────────

@app.route('/zones', methods=['GET'])
def get_zones():
    """Return all zones ranked by priority (highest score first)."""
    with zones_lock:
        sorted_zones = sorted(zones_db.values(), key=lambda z: z['score'], reverse=True)
        for i, zone in enumerate(sorted_zones):
            zone['priority'] = i + 1
        return jsonify({"zones": sorted_zones})


@app.route('/zones/override', methods=['POST'])
def override_zone():
    """Manual override for a zone's action or color."""
    data      = request.get_json()
    zone_id   = data.get('zone_id')
    new_action = data.get('action')
    new_color  = data.get('color')

    with zones_lock:
        if zone_id not in zones_db:
            return jsonify({"error": "Zone not found"}), 404
        if new_action:
            zones_db[zone_id]['override'] = {
                "original_action": zones_db[zone_id]['action'],
                "new_action":      new_action,
                "overridden_at":   datetime.now(timezone.utc).isoformat()
            }
            zones_db[zone_id]['action'] = new_action
        if new_color and new_color in ('red', 'yellow', 'green'):
            zones_db[zone_id]['color'] = new_color
        return jsonify({"status": "success", "zone": zones_db[zone_id]})


@app.route('/zones/clear', methods=['POST'])
def clear_zones():
    """Clear all zones from memory."""
    with zones_lock:
        zones_db.clear()
    return jsonify({"status": "cleared"})


# ── Static file serving ───────────────────────────────────────────────────────

@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Explicitly serve annotated result files (images / videos)."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Robo Xavier Pi — Starting on http://0.0.0.0:8000")
    print(f"📡 GPS NEO-6M reader active on {GPS_SERIAL_PORT}")
    # debug=False and threaded=True are important on Pi:
    #   debug=False  → no reloader (avoids double-spawning the GPS + live-worker threads)
    #   threaded=True → each Flask request gets its own thread
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
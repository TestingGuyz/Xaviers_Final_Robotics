# ⚡ Robo Xavier — AI Disaster Response Dispatch System

A field-deployable, multi-model AI platform for real-time disaster scene analysis, zone prioritization, and emergency dispatch. Built for Raspberry Pi (field unit) and standard servers/laptops (command unit), with a unified browser-based dispatch hub.

---

## 🗂 Project Structure

```
robo-xavier/
├── app.py                  # Main server (laptop/server — full GPU support)
├── app_pi.py               # Raspberry Pi server (optimized, GPS NEO-6M support)
├── templates/
│   ├── index.html          # Main Hub UI (upload + live feed + dispatch)
│   └── index_pi.html       # Pi-specific UI (simplified, GPS-aware)
├── static/
│   ├── uploads/            # Temporary frame/image storage
│   └── results/            # Annotated output images and videos
├── .env                    # API keys and configuration (see setup below)
└── requirements.txt        # Python dependencies
```

---

## 🧠 AI Models

| # | Model | Source | Detects |
|---|-------|--------|---------|
| 1 | **xView2** | Colab remote | Building & structural damage |
| 2 | **YOLOv8 Fire & Smoke** | Colab remote / local weights | Fire, flames, smoke, haze |
| 3 | **LADI-v2** | Colab remote | Scene-level disaster classification |
| 4 | **Ambulance Detector** | Roboflow API | Emergency vehicles |
| 5 | **Safety Vest Detector** | Roboflow API | Rescue personnel identification |
| 6 | **Groq Vision AI** | Groq API (LLaMA 4 Scout) | Holistic scene verification + summary |

All models run in parallel via a `ThreadPoolExecutor`. Model 2 falls back gracefully from remote → local weights if the Colab server is unreachable.

---

## 🚀 Features

- **Image & Video Analysis** — Upload aerial/drone images or video clips for full multi-model inference
- **Live Feed Mode** — Real-time webcam streaming with async inference (latest-frame-wins architecture — no frame queue buildup)
- **Zone Scoring Engine** — Weighted damage scoring across 6 categories: building damage, road damage, fire, smoke, flood, debris
- **Dispatch Console** — Auto-prioritized zone cards with recommended actions, color-coded severity (🔴🟡🟢), and manual override support
- **OpenStreetMap Integration** — All analyzed zones plotted on a live Leaflet map with color-coded markers
- **GPS Support**:
  - 🛰️ **Hardware** — NEO-6M GPS module via UART (Raspberry Pi)
  - 📍 **Browser** — Geolocation API fallback when hardware GPS is unavailable
  - ✏️ **Manual** — Coordinate input via the UI
- **Rolling Situational Context** — Groq summary model receives the last 3 scene summaries for coherent situational awareness across frames

---

## 🖥️ System Requirements

### Main Server (`app.py`)
- Python 3.9+
- Machine with internet access (for Roboflow + Groq APIs)
- Optional: CUDA GPU for local YOLOv8 inference

### Raspberry Pi (`app_pi.py`)
- Raspberry Pi 4 or 5 recommended
- Raspberry Pi OS (64-bit)
- NEO-6M GPS module connected via UART (`/dev/serial0`)
- Python 3.9+
- `pyserial` installed

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-org/robo-xavier.git
cd robo-xavier
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
# API Keys
ROBOFLOW_API_KEY=your_roboflow_key
GROQ_API_KEY=your_groq_key

# Remote Colab Model URLs (leave blank if not using)
COLAB_MODEL_1_URL=https://your-colab-ngrok-url-model1
COLAB_MODEL_2_URL=https://your-colab-ngrok-url-model2
COLAB_MODEL_3_URL=https://your-colab-ngrok-url-model3

# Local Model 2 Path (optional — for offline fire/smoke detection)
MODEL_2_PATH=/path/to/your/best.pt

# Raspberry Pi GPS (only needed for app_pi.py)
GPS_SERIAL_PORT=/dev/serial0
GPS_BAUD_RATE=9600
```

### 3. Run

**Main server (laptop/desktop):**
```bash
python app.py
# → http://localhost:8000
```

**Raspberry Pi server:**
```bash
python app_pi.py
# → http://0.0.0.0:8000  (accessible on local network)
```

> ⚠️ On Pi, run with `debug=False` (already set). The `threaded=True` flag ensures the GPS reader and live-feed worker threads are not double-spawned by Flask's reloader.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dispatch hub UI |
| `POST` | `/process` | Upload image or video for inference |
| `GET` | `/gps` | Get current GPS coordinates and source |
| `POST` | `/gps/browser` | Push browser geolocation to server |
| `POST` | `/live_push` | Push a webcam frame for async inference |
| `GET` | `/live_status` | Poll for the latest live inference result |
| `GET` | `/zones` | Get all zones sorted by priority score |
| `POST` | `/zones/override` | Override a zone's action or severity color |
| `POST` | `/zones/clear` | Clear all zones from memory |

### `/process` — Request Body (multipart/form-data)

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Image (jpg/png) or video (mp4/avi/mov/webm) |
| `models` | JSON string or `"all"` | List of model IDs to run |
| `latitude` | string | Manual GPS latitude |
| `longitude` | string | Manual GPS longitude |
| `url_model1` | string | Override Colab URL for Model 1 |
| `url_model2` | string | Override Colab URL for Model 2 |
| `url_model3` | string | Override Colab URL for Model 3 |

### `/live_push` — Request Body (JSON)

```json
{
  "frame": "data:image/jpeg;base64,...",
  "lat": 28.6139,
  "lon": 77.2090,
  "models": ["model1", "model2", "model6"],
  "url_model1": "",
  "url_model2": "",
  "url_model3": ""
}
```

---

## 🏗️ Live Feed Architecture

The live feed uses a **"latest-frame wins"** single-slot buffer to avoid stale queue buildup when inference is slower than the capture rate.

```
Browser (every 2s)          Backend                    Browser Poll (every 2s)
─────────────────          ─────────────────          ─────────────────────────
push frame ─────────────▶  _live_slot (overwrites)
push frame ─────────────▶  _live_slot (overwrites)    poll /live_status ──▶ {}
push frame ─────────────▶  _live_slot (overwrites)
                            ↓  worker wakes up
                           runs inference (~10s)
                            ↓  stores result           poll /live_status ──▶ {zone, detections, ...}
push frame ─────────────▶  _live_slot (newest only)
```

Frames that arrive while the worker is busy are silently discarded. The `frames_dropped` counter in `/live_status` reports how many were dropped.

---

## 🗺️ Zone Scoring

Each analyzed zone receives a weighted damage score:

| Category | Weight |
|----------|--------|
| Building Damage | 35% |
| Road Damage | 25% |
| Fire | 20% |
| Flood | 10% |
| Smoke | 5% |
| Debris | 5% |

**Zone color logic:**
- 🔴 **Red** — Building damage detected, or road damage with score > 0.4
- 🟡 **Yellow** — Road damage only
- 🟢 **Green** — No significant damage categories detected

**Suggested actions** are automatically generated based on color + detected categories and can be manually overridden per zone via the Dispatch Console.

---

## 🛰️ GPS Priority (Raspberry Pi)

The Pi server uses a three-tier GPS fallback chain:

1. **Hardware NEO-6M** (highest priority) — parsed from NMEA `$GPGGA` / `$GNGGA` sentences over UART
2. **Browser Geolocation API** — accepted only when no hardware fix is active
3. **Manual input** — coordinate fields in the UI

The hardware GPS reader runs as a persistent daemon thread and automatically reconnects on serial failure.

---

## 📦 Dependencies

```
flask
python-dotenv
opencv-python
numpy
requests
ultralytics      # YOLOv8 local inference (optional)
pyserial         # NEO-6M GPS on Raspberry Pi (optional)
```

Frontend uses CDN-hosted libraries — no build step required:
- [Leaflet.js](https://leafletjs.com/) — OpenStreetMap
- [Inter](https://fonts.google.com/specimen/Inter) — UI font

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push and open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.

---

*Built for rapid field deployment in disaster response scenarios. Designed to run on Raspberry Pi 4/5 with degraded-gracefully architecture — every component is optional and the system remains functional when external APIs or hardware are unavailable.*

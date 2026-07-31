# LightningExtractor

Automatic lightning detection, AI enhancement, and clean trail extraction from
nighttime videos.

## Pipeline

```
input/*.mp4
   │  1. main.py             detection + clip export + reels + PNG isolation
   ▼
output/<video>/clips/
   │  2. enhance_videos.py   denoise → RIFE 2x → lightning protection → NVENC H.265
   ▼
enhanced_output/<video>/clips/
   │  3. extract_trail.py    background removal + top-hat MIP → channel-only stills
   ▼
bolt_output/<video>/strike_NNN_trail.png   (+ _mip.png, contact sheets, report.csv)
```

### Stage 1 — Detection (`main.py`)

Scans the input videos, detects lightning flashes, exports clips, selects the
sharpest frame per event, and builds per-video reports, reels, transparent PNG
isolations, and contact sheets.

### Stage 2 — AI enhancement (`enhance_videos.py`)

Per clip:
1. **Denoise** via FFmpeg `nlmeans_vulkan` and extract frames
2. **RIFE 2x interpolation** — doubles the frame rate for smooth slow-motion
   (original camera frames sit at *even* frame indices; odd indices are
   AI-interpolated)
3. **Lightning protection** — re-synthesizes each interpolated frame so the
   strike itself is never an AI artifact (the bright channel is blended back
   from the real camera frames)
4. **NVENC H.265 encode** at the doubled fps
5. **Side-by-side comparison** video (original vs enhanced), merged per video

### Stage 3 — Trail extraction (`extract_trail.py`)

Isolates the lightning channel only (no fog) as high-quality stills:

1. **Background estimate** — per-pixel median of the darkest frames (static
   night scene)
2. **Flash frames** — frames significantly brighter than the background
3. **Delta MIP** — per-pixel maximum of `(flash − background)` across flash
   frames, recovering every branch while removing the static sky
4. **Grayscale top-hat** — `trail = delta_MIP − opening(delta_MIP)`. The
   opening (disk kernel wider than the channel) leaves only the broad fog
   envelope; subtracting it keeps the full channel (including its saturated
   interior) and removes the diffuse fog
5. **Threshold + cleanup** — 3σ threshold, speckle removal, small close to
   bridge channel gaps, drop components under a minimum area
6. **Render** — the channel in its natural colors on a pure black background

A report (`bolt_output/report.csv`) records per-clip quality metrics:
`flash_frames`, `trail_area_pct`, `components`, `largest_px`, `mean_brightness`.

## Installation

```bash
pip install -r requirements.txt
```

Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org and add to PATH, or place `ffmpeg.exe` in the project directory
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

RIFE/Real-ESRGAN Vulkan binaries live under `tools/rife` and `tools/realesrgan`
and must be present for Stage 2.

## Usage

### Detection

```bash
# Process all videos in input/
python main.py

# Slow-motion (4x slow = 0.25 factor)
python main.py --slowmo 0.25

# Parallel processing / GPU acceleration
python main.py --parallel --use-gpu

# Compile highlight reel of all detected events
python main.py --compile --slowmo 0.25

# Single video with custom confidence
python main.py --input path/to/video.mp4 --confidence 0.6

# Skip PNG isolation and contact sheets
python main.py --no-png --no-contact
```

### Enhancement

```bash
# Enhance all detected clips
python enhance_videos.py

# Specific videos, or multiple at once
python enhance_videos.py --videos VID_20250828_193327 VID_20250902_200200

# Parallel clip processing
python enhance_videos.py --max-workers 2
```

### Trail extraction

```bash
# All enhanced clips
python extract_trail.py

# Specific videos
python extract_trail.py --videos VID_20250828_193327

# Tuning (defaults shown)
python extract_trail.py --disk-r 25 --thresh-mult 3.0 --min-area 150 --close-kernel 9
```

### Contact sheets

```bash
# Grid of all extracted trails per video, labeled with size + component count
python make_contact_sheet.py
```

## Output Structure

```
output/
├── video_name/
│   ├── timestamps.txt           # Timestamps of each strike
│   ├── report.json              # Full detection report
│   ├── clips/                   # Extracted video clips
│   ├── frames/                  # Sharpest frame from each clip
│   ├── lightning_png/           # Transparent PNG isolations
│   ├── contact_sheet.png        # Grid of all detected frames
│   └── *_reel.mp4               # Per-video highlight reel

enhanced_output/
├── video_name/
│   ├── clips/                   # RIFE 2x enhanced clips (60 fps)
│   ├── comparison/              # Side-by-side original-vs-enhanced clips
│   ├── *_comparison.mp4         # Merged comparison video
│   ├── *_reel.mp4 / *_contact_sheet.png   # Copied from output/
│   └── report.json

bolt_output/
├── video_name/
│   ├── strike_NNN_trail.png     # Lightning channel only (natural color, black bg)
│   └── strike_NNN_mip.png       # Reference: full delta-MIP frame
├── <video>_contact.png          # Review sheet of all trails in the video
└── report.csv                   # Quality metrics per clip
```

## Configuration

### Detection (`main.py`)

Edit `config.py` or use CLI flags:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence` | 0.5 | Detection confidence threshold |
| `--slowmo` | None | Slow-motion factor (e.g. 0.25) |
| `--step` | 2 | Process every Nth frame |
| `--use-gpu` | False | Enable GPU acceleration |
| `--parallel` | False | Process videos in parallel |
| `--compile` | False | Compile highlight reel |

### Trail extraction (`extract_trail.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--videos` | all | Video names to process |
| `--disk-r` | 25 | Top-hat opening disk radius (wider than the channel) |
| `--thresh-mult` | 3.0 | Top-hat threshold in multiples of its std |
| `--min-area` | 150 | Drop connected components smaller than this |
| `--close-kernel` | 9 | Closing kernel size to bridge channel gaps |

## Current Status

- **Stage 1 & 2 complete**: 172 events detected and enhanced across 6 source
  videos (landscape 1920×1080 and portrait 1080×1920).
- **Stage 3 complete**: all 172 trails extracted with the top-hat pipeline and
  reviewed via contact sheets (`bolt_output/*_contact.png`).
- **Pending validation**: visual check that trails read as continuous channels
  rather than fragmented pieces.
- **Evaluated, not shipped**: region-growing (dilating the 3σ seed through a
  ~1.5–2σ mask) bridges dim gaps and connects large bolts, but is inconsistent
  across clips and risks leaking fog back in. If fragmentation is confirmed
  visually, add it behind a flag (e.g. `--bridge`) before finalizing.

## How It Works

1. **Pre-scan calibration** - Samples first N frames for per-video baseline statistics
2. **Multi-signal detection** - Combines brightness spike, bright pixel ratio, and edge increase signals
3. **Duplicate merging** - Groups nearby detections within 0.5s window, keeps highest confidence
4. **Clip export** - Extracts video segments around each detection via FFmpeg
5. **Sharpest frame** - Selects frame with highest Laplacian variance from each clip
6. **Enhancement** - Denoise, RIFE 2x interpolation, lightning protection, NVENC encode
7. **Trail extraction** - Median background removal, delta-MIP, grayscale top-hat, cleanup

## Superseded Tools

- `extract_bolt.py` — exports the literal camera frame per event (keeps the fog).
  Replaced by `extract_trail.py`.
- `lightning_isolator.py` — masked black-background bolt renders. Rejected in
  favor of natural-color trails.

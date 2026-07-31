# LightningExtractor

Automatic lightning detection and extraction from nighttime videos.

## Installation

```bash
pip install -r requirements.txt
```

Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org and add to PATH, or place `ffmpeg.exe` in the project directory
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

## Usage

```bash
# Process all videos in input/
python main.py

# Process with slow-motion (4x slow = 0.25 factor)
python main.py --slowmo 0.25

# Process in parallel
python main.py --parallel

# Use GPU acceleration (if available)
python main.py --use-gpu

# Compile highlight reel of all detected events
python main.py --compile --slowmo 0.25

# Process a single video with custom confidence
python main.py --input path/to/video.mp4 --confidence 0.6

# Skip PNG generation and contact sheets
python main.py --no-png --no-contact
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
│   └── contact_sheet.png        # Grid of all detected frames
├── highlight_reel.mp4           # (optional) compiled compilation
```

## Configuration

Edit `config.py` or use CLI flags to adjust detection parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence` | 0.5 | Detection confidence threshold |
| `--slowmo` | None | Slow-motion factor (e.g. 0.25) |
| `--step` | 2 | Process every Nth frame |
| `--use-gpu` | False | Enable GPU acceleration |
| `--parallel` | False | Process videos in parallel |
| `--compile` | False | Compile highlight reel |

## How It Works

1. **Pre-scan calibration** - Samples first N frames for per-video baseline statistics
2. **Multi-signal detection** - Combines brightness spike, bright pixel ratio, and edge increase signals
3. **Duplicate merging** - Groups nearby detections within 0.5s window, keeps highest confidence
4. **Clip export** - Extracts video segments around each detection via FFmpeg
5. **Sharpest frame** - Selects frame with highest Laplacian variance from each clip
6. **Lightning isolation** - Produces transparent PNGs with background removed

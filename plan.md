# Project Prompt: Automatic Lightning Detection and Extraction from Night Videos

You are an expert Computer Vision and Python engineer. Your task is to design and implement a production-quality Python application that automatically detects lightning events in long nighttime videos, extracts them, and generates high-quality outputs.

## Objective

Develop a modular, well-documented, efficient, and extensible Python application capable of processing multiple videos automatically.

The application must:

* Scan every video inside an `input/` directory.
* Detect lightning strikes automatically.
* Save precise timestamps.
* Export a short video clip around each detected lightning strike.
* Extract the sharpest frame from each lightning clip.
* Optionally generate transparent PNG images containing only the lightning.
* Produce logs and statistics for every processed video.

The application should be robust enough to process many hours of footage with minimal user intervention.

---

# Project Structure

Implement the following architecture:

```text
LightningExtractor/

│
├── input/
│      video1.mp4
│      video2.mp4
│      ...
│
├── output/
│      video1/
│          timestamps.txt
│          report.json
│          clips/
│          frames/
│          lightning_png/
│
├── config.py
├── main.py
│
├── video_scanner.py
├── lightning_detector.py
├── clip_exporter.py
├── frame_selector.py
├── lightning_isolator.py
│
├── utils.py
├── logger.py
│
├── requirements.txt
└── README.md
```

Each module must have a single responsibility.

Avoid writing one large script.

---

# Required Technologies

Use:

* Python 3.12+
* OpenCV
* NumPy
* SciPy
* tqdm
* FFmpeg (called from Python)
* pathlib
* dataclasses
* typing
* logging

Use only stable and widely adopted libraries.

---

# Processing Pipeline

Implement the following pipeline:

Video

↓

Read frame-by-frame

↓

Convert to grayscale

↓

Compute detection features

↓

Detect candidate lightning events

↓

Merge duplicate detections

↓

Save timestamps

↓

Export surrounding clips

↓

Select sharpest frame

↓

Generate transparent PNG (optional)

↓

Generate report

---

# Lightning Detection

Do NOT rely on only average brightness.

Instead, combine multiple complementary signals.

## Signal 1 — Brightness Increase

Compute:

* Mean grayscale intensity
* Difference from previous frame
* Rolling average
* Rolling standard deviation

Use adaptive thresholds rather than fixed values.

---

## Signal 2 — Percentage of Bright Pixels

For every frame compute:

* pixels >200
* pixels >220
* pixels >240

Calculate:

bright_pixel_ratio = bright_pixels / total_pixels

This helps distinguish lightning from gradual exposure changes.

---

## Signal 3 — Edge Increase

Lightning creates many thin high-frequency edges.

Compute:

* Canny edges
* Edge pixel count
* Difference with previous frame

A sudden increase is a strong lightning indicator.

---

## Signal Fusion

Create a confidence score using all signals.

Example:

confidence =
0.45 * brightness_score +
0.35 * bright_pixel_score +
0.20 * edge_score

The weights should be configurable.

Only accept detections above a configurable confidence threshold.

---

# Duplicate Removal

Lightning usually lasts several consecutive frames.

Merge detections occurring within a configurable time window (e.g., 0.5 seconds) into a single event.

Keep the frame with the highest confidence as the representative event.

---

# Timestamp Export

Generate:

timestamps.txt

Example:

```text
00:01:42.350
00:04:11.120
00:07:56.770
```

Also generate a JSON report containing:

* frame number
* timestamp
* confidence
* brightness score
* edge score
* bright pixel ratio

---

# Clip Export

Use FFmpeg instead of OpenCV for exporting clips.

Export:

1 second before detection

*

1 second after detection

Default clip duration:

2 seconds

Make this configurable.

---

# Sharpest Frame Selection

From every exported clip:

Compute the Laplacian variance for every frame.

Select the frame with the highest sharpness score.

Save:

```
strike_001.png
```

Store the sharpness value in the report.

---

# Lightning Isolation

Generate transparent PNGs.

Pipeline:

Convert to grayscale

↓

Threshold

↓

Morphological opening

↓

Morphological closing

↓

Connected components

↓

Remove tiny components

↓

Keep lightning structures

↓

Create alpha channel

↓

Save transparent PNG

The output should preserve the lightning while removing as much background as possible.

---

# Performance Requirements

The program should efficiently process long videos.

Requirements:

* Sequential video reading (avoid loading entire videos into memory)
* Low memory usage
* Progress bars using tqdm
* Logging for every stage
* Graceful handling of corrupted videos

---

# Configuration

All thresholds must be configurable in `config.py`.

Example parameters:

* brightness threshold
* edge threshold
* confidence threshold
* merge interval
* clip duration
* threshold values
* output format
* FFmpeg path

No hardcoded magic numbers inside the code.

---

# Logging

Create a logger.

Example:

```text
Processing video1.mp4...

Frames scanned: 52,134

Lightning events detected: 14

Exporting clips...

Selecting sharpest frames...

Generating PNGs...

Completed successfully.
```

---

# Error Handling

Handle:

* missing FFmpeg
* unreadable videos
* unsupported codecs
* empty folders
* interrupted execution

The application must never crash unexpectedly.

---

# Documentation

Generate:

* README.md
* installation guide
* project architecture
* configuration guide
* execution examples
* troubleshooting section

Document every public function using Google-style docstrings.

Use full type hints throughout the project.

---

# Code Quality

The code must:

* follow PEP 8
* use dataclasses where appropriate
* avoid duplicated code
* separate concerns cleanly
* be maintainable and extensible
* include descriptive variable names
* use pathlib instead of os.path where practical

---

# Bonus Features (Optional)

If possible, also implement:

* Automatic generation of contact sheets showing all detected lightning frames.
* CSV summary of detections.
* HTML report with thumbnails and timestamps.
* Automatic extraction of the single brightest lightning event.
* Ability to process nested input folders recursively.
* Optional GPU acceleration using OpenCV CUDA when available.
* Command-line interface (CLI) using argparse.
* Unit tests for the detection logic.
* Sample videos and expected outputs for testing.

---

# Deliverables

Provide the complete project with:

1. All source code files.
2. `requirements.txt`.
3. `README.md`.
4. A clear explanation of the processing pipeline.
5. Instructions to install FFmpeg.
6. Example commands to run the application.
7. Suggestions for improving detection accuracy on difficult nighttime videos.

The final solution should be production-quality, modular, efficient, thoroughly documented, and suitable for processing many hours of nighttime storm footage with minimal manual intervention.

## Road-Scene Object Detection (YOLOv8)

This project detects objects in **road scenes** (cars, pedestrians, traffic elements, etc.) using **YOLOv8**. 

It supports:
- **Images**
- **Videos**
- **Webcam / camera streams**

All of this is driven from a single script: `detect_road.py`.

---

## 1. Requirements

### 1.1. Python

- Python 3.8+ (recommended)

Check your Python version:

```bash
python --version
```

### 1.2. Install dependencies

From the project folder (where `detect_road.py` lives):

```bash
pip install ultralytics opencv-python numpy tk
```

> Note: On some systems, `tkinter` (used for the file browser) comes with Python. If you see an error about `tkinter`, install it using your OS/Python installer.

---

## 2. Basic Usage

All commands are run from the project directory:

```bash
cd path/to/project
python detect_road.py [OPTIONS]
```

Key options are:

- `--source` – what to run detection on (webcam, image, video, or `browse` dialog)
- `--show` – show a live window with detections
- `--save` – save annotated outputs under `runs/detect/...`

### 2.1. Run on webcam

Use your default webcam (index `0`):

```bash
python detect_road.py --source 0 --show --save
```

- Opens a window showing real‑time detections.
- Saves annotated frames/video under `runs/detect/...`.

If you have multiple cameras, try `--source 1`, `--source 2`, etc.

### 2.2. Run on image/video via file browser

Open a file‑selection dialog and choose an image or video:

```bash
python detect_road.py --source browse --show --save
```

- A dialog will appear so you can select `.jpg`, `.png`, `.mp4`, `.avi`, etc.
- After choosing the file, the script runs detection and shows/saves results.

### 2.3. Run on a specific file path

You can also pass a direct path instead of using the browser:

```bash
python detect_road.py --source path/to/your_video.mp4 --show --save
```

or for an image:

```bash
python detect_road.py --source path/to/image.jpg --show --save
```

---

## 3. Commands and What They Do

Below is a summary of the main options supported by `detect_road.py`.

### 3.1. Core options

- `--source <value>`
  - **`0`, `1`, ...** – webcam index (0 = default camera)
  - **`browse`** – open a file selection dialog (image/video)
  - **`path/to/file`** – image or video file path
  - **`path/to/folder`** – directory with images/videos
  - **URL/stream** – RTSP/RTMP/HTTP/HTTPS stream URL

- `--show`
  - If present, opens an OpenCV window showing annotated frames.
  - Press `Esc` or `q` in the window to stop.

- `--save`
  - Save annotated outputs using Ultralytics’ default structure.
  - Results go under `runs/detect/predict`, `predict2`, etc.

### 3.2. Model and inference options

- `--model <path>`
  - YOLOv8 model weights (default: `yolov8n.pt`).
  - You can swap with `yolov8s.pt`, `yolov8m.pt`, a custom trained model, etc.

- `--imgsz <int>`
  - Inference image size (default: `640`).
  - Larger values can improve accuracy but are slower.

- `--conf <float>`
  - Confidence threshold for detections (default: `0.25`).
  - Increase to reduce low‑confidence boxes.

- `--device <str>`
  - Compute device: e.g. `cpu`, `cuda`, `0`, `0,1`.
  - Default (`None`) lets Ultralytics choose automatically.

### 3.3. Output and saving options

- `--project <path>`
  - Base directory for saving results (default: `runs/detect`).

- `--name <run_name>`
  - Sub‑folder name inside `--project`.
  - If omitted, the script auto‑increments: `predict`, `predict2`, ...

- `--save-mp4-direct`
  - Instead of the default saving, directly writes annotated video to MP4.
  - Output file is typically `runs/detect/<run_name or auto>/0.mp4`.

- `--fps <float>`
  - Output FPS when using `--save-mp4-direct` (default: `25.0`).

- `--reencode-mp4`
  - After running, search for saved `.avi` files and convert them to `.mp4`.
  - Helpful for broader video‑player compatibility.

- `--delete-avi`
  - Used together with `--reencode-mp4`.
  - Deletes the original `.avi` after successful MP4 conversion.

---

## 4. Example Commands

### 4.1. Webcam, show only

```bash
python detect_road.py --source 0 --show
```

### 4.2. Webcam, show and save annotated video as MP4 directly

```bash
python detect_road.py --source 0 --show --save-mp4-direct --fps 30
```

### 4.3. Browse for image/video, show and save 

```bash
python detect_road.py --source browse --show --save
```

### 4.4. Use a custom YOLO model on a video file

```bash
python detect_road.py --source path/to/road_video.mp4 --model path/to/custom_yolov8.pt --show --save
```

### 4.5. Convert saved AVI outputs to MP4 and delete AVI

After a normal run with `--save`, you can re‑encode like this:

```bash
python detect_road.py --source path/to/road_video.mp4 --save --reencode-mp4 --delete-avi
```

This will:
- Run detection
- Convert any generated `.avi` files to `.mp4`
- Remove the original `.avi` files (if conversion succeeded)

---

## 5. Output Location

- All results are saved under the `--project` directory (default: `runs/detect`).
- Folders are auto‑incremented like:
  - `runs/detect/predict`
  - `runs/detect/predict2`
  - `runs/detect/predict3`
- When using `--save-mp4-direct`, you’ll typically find output at:
  - `runs/detect/<run_name or auto>/0.mp4`

---

## 6. Troubleshooting

- **No frames processed / camera not working**
  - Check that no other app is using the webcam.
  - Try `--source 1` (or 2, 3, ...) if you have multiple cameras.
  - Ensure the correct path or URL when not using webcam.

- **Window does not appear**
  - Make sure you passed `--show`.
  - Some remote/virtual environments may block GUI windows.

- **Tk / file dialog errors**
  - Ensure `tkinter` is installed with your Python.
  - If the dialog still fails, use direct file paths instead of `--source browse`.

If you run into any other issues or want to extend the project, you can review `detect_road.py` and customize the logic for your use case.
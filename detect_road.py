import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 road-scene detection: image/video/webcam")
    parser.add_argument("--source", type=str, default="0", help="Path to image/video, directory, or webcam index (e.g., 0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or specific device index")
    parser.add_argument("--show", action="store_true", help="Display annotated frames in a window")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs to runs/detect")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default=None, help="Run name (auto-increments if omitted)")
    parser.add_argument("--reencode-mp4", action="store_true", help="Convert saved .avi files to .mp4 for compatibility")
    parser.add_argument("--delete-avi", action="store_true", help="Delete .avi after successful re-encode to .mp4")
    parser.add_argument("--save-mp4-direct", action="store_true", help="Write annotated video directly to MP4 (no AVI)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output FPS when using --save-mp4-direct")
    return parser.parse_args()


def is_webcam_source(src: str) -> bool:
    if src.isdigit():
        return True
    if src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return True
    return False


def browse_source() -> str:
    """Open a file dialog and return the selected path, or empty string if canceled."""
    # Hide the root Tk window
    root = Tk()
    root.withdraw()
    root.update()
    filetypes = [
        ("Media files", "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
        ("Images", "*.jpg *.jpeg *.png *.bmp *.gif"),
        ("Videos", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Select image or video", filetypes=filetypes)
    root.destroy()
    return path or ""


def next_run_dir(project: Path) -> Path:
    """Mimic Ultralytics auto-increment (predict, predict2, ...) for a project folder."""
    base = project / "predict"
    if not base.exists():
        return base
    i = 2
    while True:
        d = project / f"predict{i}"
        if not d.exists():
            return d
        i += 1


def main():
    args = parse_args()

    model = YOLO(args.model)

    source = args.source
    if isinstance(source, str) and source.lower() == "browse":
        picked = browse_source()
        if not picked:
            print("No file selected. Exiting.")
            return
        source = picked
    if source.isdigit():
        source_in = int(source)
    else:
        source_in = source

    stats_counts = defaultdict(int)
    stats_conf_sum = defaultdict(float)
    total_frames = 0
    total_detections = 0
    last_save_dir: Path | None = None
    writer = None
    direct_out_path: Path | None = None

    predict_kwargs = dict(
        source=source_in,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        save=(args.save and not args.save_mp4_direct),
        project=args.project,
    )
    if args.name is not None:
        predict_kwargs["name"] = args.name

    try:
        results_gen = model.predict(**predict_kwargs)
        for res in results_gen:
            total_frames += 1
            plot_img = res.plot()

            # Track save directory from Ultralytics results (set once available)
            try:
                if hasattr(res, "save_dir") and res.save_dir is not None:
                    last_save_dir = Path(res.save_dir)
            except Exception:
                pass

            # Direct MP4 saving: initialize once with frame size
            if args.save_mp4_direct:
                if writer is None:
                    # Determine save directory
                    project_dir = Path(args.project)
                    run_dir = (project_dir / args.name) if args.name else next_run_dir(project_dir)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    direct_out_path = run_dir / "0.mp4"
                    h, w = plot_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(direct_out_path), fourcc, float(args.fps), (w, h))
                    print(f"Writing direct MP4 to: {direct_out_path}")
                writer.write(plot_img)

            if res.boxes is not None and len(res.boxes) > 0:
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                names = res.names
                for c, p in zip(cls_ids, confs):
                    cls_name = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
                    stats_counts[cls_name] += 1
                    stats_conf_sum[cls_name] += float(p)
                    total_detections += 1

            if args.show:
                try:
                    cv2.imshow("YOLOv8 Detect", plot_img)
                    if cv2.waitKey(1 if is_webcam_source(str(source_in)) else 10) & 0xFF in (27, ord('q')):
                        break
                except cv2.error:
                    pass

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

    if total_frames == 0:
        print("No frames processed. Check your source path or camera index.")
        return

    print("\n===== Detection Summary =====")
    print(f"Frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    if total_detections == 0:
        print("No objects detected above the confidence threshold.")
    else:
        print("Per-class counts and avg confidence:")
        for cls_name in sorted(stats_counts.keys()):
            count = stats_counts[cls_name]
            avg_conf = stats_conf_sum[cls_name] / max(count, 1)
            print(f"- {cls_name}: {count} (avg conf {avg_conf:.3f})")

    if args.save or args.save_mp4_direct:
        proj = Path(args.project)
        print(f"\nSaved annotated outputs under: {proj.resolve()} (Auto-created by Ultralytics)")

        # Optional post-process: convert any .avi files to .mp4 for broader compatibility
        if args.reencode_mp4:
            # Prefer the actual save dir if provided by results; otherwise fall back to project
            save_root = last_save_dir if last_save_dir is not None else proj
            converted = 0
            for avi_path in save_root.rglob("*.avi"):
                mp4_path = avi_path.with_suffix(".mp4")
                try:
                    cap = cv2.VideoCapture(str(avi_path))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if width == 0 or height == 0:
                        cap.release()
                        continue
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                    writer.release()
                    cap.release()
                    print(f"Re-encoded to: {mp4_path}")
                    converted += 1
                    if args.delete_avi:
                        try:
                            avi_path.unlink(missing_ok=True)
                            print(f"Deleted source AVI: {avi_path}")
                        except Exception as e:
                            print(f"Could not delete {avi_path}: {e}")
                except Exception as e:
                    print(f"Failed to re-encode {avi_path}: {e}")
            if converted == 0:
                print("No .avi files found to convert, or conversion not needed.")


if __name__ == "__main__":
    sys.exit(main())

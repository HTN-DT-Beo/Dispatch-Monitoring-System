# Sources/Pipeline/infer.py
import numpy as np
from ultralytics import YOLO
from boxmot import BoTSORT
import cv2, onnxruntime as ort
from pathlib import Path
import torch

# ─── Đường dẫn ───────────────────────────────────────────────
MODEL_PATH = "Models"
CLASSIFICATION_DIR = MODEL_PATH + "/Classification"
DETECTION_DIR = MODEL_PATH + "/Detection"
VIDEO_IN  = Path(r"D:\Project\Dispatch_Monitoring_System\Dispatch-Monitoring-System\Data\test_04.mp4")
VIDEO_OUT = "output.mp4"

# ─── Tham số ─────────────────────────────────────────────────
IMG_SIZE = 224  # chuẩn EfficientNet-B0
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

LABELS_DISH = ["empty", "kakigori", "not_empty"]
LABELS_TRAY = ["empty", "kakigori", "not_empty"]  # sửa lại nếu nhãn khác

# ─── Hàm phân loại ───────────────────────────────────────────
def classify(img, sess, labels):
    if img.size == 0:
        return "unk"
    im = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    im = ((im - MEAN) / STD).transpose(2, 0, 1)[None]  # (1,3,H,W)
    logits = sess.run(None, {"images": im})[0]
    return labels[int(np.argmax(logits, 1)[0])]

# ─── Load mô hình ────────────────────────────────────────────
det_model = YOLO(f"{DETECTION_DIR}/best.pt")
dish_sess = ort.InferenceSession(f"{CLASSIFICATION_DIR}/dish_cls.onnx", providers=["CPUExecutionProvider"])
tray_sess = ort.InferenceSession(f"{CLASSIFICATION_DIR}/tray_cls.onnx", providers=["CPUExecutionProvider"])

# ─── Hàm chạy pipeline ──────────────────────────────────────
def run(video_in: Path, video_out: str = "videos/output.mp4", show: bool = True):
    writer = None
    with torch.no_grad():
        frame_count = 0
        for res in det_model.track(
            source=str(video_in),
            tracker="botsort.yaml",
            classes=[0, 1],
            conf=0.15,
            stream=True,
            show=show,
            verbose=False,
        ):
            frame = res.orig_img
            if frame is None:
                print("⚠️ Không có frame")
                continue
            if writer is None:
                h, w = frame.shape[:2]
                fps = res.fps if hasattr(res, "fps") else 30
                fourcc = cv2.VideoWriter_fourcc(*"mp4v") #avc1
                writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))
                if not writer.isOpened():                       # ➊
                    raise RuntimeError("❌ VideoWriter mở không thành công")
                print("🎬 Writer opened:", video_out)            # ➋

            # ── Track & phân loại (giữ nguyên) ──
            if res.boxes.id is not None:
                for box, tid, cid in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.id.cpu().numpy(),
                    res.boxes.cls.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]

                    if cid == 0:
                        state = classify(crop, dish_sess, LABELS_DISH)
                        label_cls = "dish"
                    else:
                        state = classify(crop, tray_sess, LABELS_TRAY)
                        label_cls = "tray"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{int(tid)}-{label_cls}-{state}"
                    cv2.putText(
                        frame,
                        text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            writer.write(frame)
            frame_count += 1

    writer.release()
    print(f"✅ Saved to {video_out}, total frames written: {frame_count}")
    print("✅ Saved to", video_out)
    return video_out

# ─── Chạy từ CLI (python infer.py in.mp4 [out.mp4]) ─────────
if __name__ == "__main__":
    import sys

    in_path  = Path(sys.argv[1]) if len(sys.argv) >= 2 else VIDEO_IN
    out_path = sys.argv[2]       if len(sys.argv) >= 3 else VIDEO_OUT
    run(in_path, out_path, show=True)
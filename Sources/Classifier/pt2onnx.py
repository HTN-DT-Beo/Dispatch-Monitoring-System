from pathlib import Path
import torch, timm

# ---- 1. Cấu hình -------------------------------------------------
IMG_SIZE   = 224          # size bạn đã dùng khi train
OPSET      = 17           # phiên bản ONNX opset
WEIGHTS_DIR = Path("Models/Classification")

# Danh sách (tên_pt, tên_onnx) cần export
EXPORTS = [
    ("dish_cls.pt", "dish_cls.onnx"),
    ("tray_cls.pt", "tray_cls.onnx"),
]

# ---- 2. Hàm build lại kiến trúc giống lúc train ------------------
def get_model(num_classes=3):
    return timm.create_model(
        "efficientnet_b0",
        pretrained=False,           # export không cần weight pretrain
        num_classes=num_classes
    )

# ---- 3. Hàm export một model -------------------------------------
def export(pt_path: Path, onnx_path: Path):
    print(f"🔄  Export {pt_path.name}  →  {onnx_path.name}")
    model = get_model()
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
    )
    print("✅ Done!")

# ---- 4. Chạy ------------------------------------------------------
if __name__ == "__main__":
    for pt_name, onnx_name in EXPORTS:
        pt_file  = WEIGHTS_DIR / pt_name
        onnx_out = WEIGHTS_DIR / onnx_name
        export(pt_file, onnx_out)
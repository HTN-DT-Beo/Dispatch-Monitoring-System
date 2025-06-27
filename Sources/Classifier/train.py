from pathlib import Path
import timm, torch, torchvision as tv
from torch import nn

# === 1. Đường dẫn cứng tới dữ liệu và nơi lưu model ==========
DISH_DATASET = Path("Data/Dataset/Classification/dish")
TRAY_DATASET = Path("Data/Dataset/Classification/tray")
MODEL_DIR    = Path("Models/Classification")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === 2. Hàm chuẩn bị DataLoader ==============================
def dataloader(root, img_size=224, bs=32):
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds = tv.datasets.ImageFolder(root, transform=tfm)
    return torch.utils.data.DataLoader(ds, bs, shuffle=True, num_workers=4), len(ds.classes)

# === 3. Hàm train một model (thêm early-stopping) =============
def train_one(root, out_path, epochs=10, bs=32, stop_loss=5e-4):
    dl, num_cls = dataloader(root, bs=bs)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_cls)
    opt   = torch.optim.AdamW(model.parameters(), 1e-4)
    lossF = nn.CrossEntropyLoss()

    for ep in range(epochs):
        for x, y in dl:
            opt.zero_grad()
            pred  = model(x)
            loss  = lossF(pred, y)
            loss.backward()
            opt.step()

        print(f"[{root.name}] epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

        # ─── Điều kiện dừng sớm ───────────────────────────────
        if loss.item() <= stop_loss:
            print(f"🔔 Early-stop: loss ≤ {stop_loss}")
            break

    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved: {out_path}")

# === 4. Training =============================================
if __name__ == "__main__":
    train_one(DISH_DATASET, MODEL_DIR / "dish_cls.pt", epochs=15)
    train_one(TRAY_DATASET, MODEL_DIR / "tray_cls.pt", epochs=15)


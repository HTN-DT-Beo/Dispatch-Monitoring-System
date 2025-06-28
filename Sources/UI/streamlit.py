import streamlit as st
import requests, tempfile, cv2
from pathlib import Path

API_ROOT = "http://dms-api:8000"
VIDEO_DIR = Path("Videos")
VIDEO_DIR.mkdir(exist_ok=True)

st.title("🎞️ Tracking Video – Streamlit")

uploaded = st.file_uploader("📂 Chọn file MP4", type=["mp4"])
if uploaded:
    st.video(uploaded)

    if st.button("🚀 Track") and uploaded:
        with st.spinner("Đang upload và xử lý…"):
            # Gửi video lên server
            resp = requests.post(
                f"{API_ROOT}/infer",
                files={"file": (uploaded.name, uploaded, "video/mp4")},
                timeout=600,
            )
            resp.raise_for_status()

            # Lưu kết quả video từ API về local
            result_bytes = resp.content
            result_path = VIDEO_DIR / f"result_{uploaded.name}"
            result_path.write_bytes(result_bytes)

        st.success("✅ Đã xử lý xong!")

        # Phát video từ đường dẫn file đã lưu
        st.subheader("🎬 Video kết quả")
        st.video(str(result_path))

        # Hiển thị frame đầu tiên
        st.subheader("🖼 Frame đầu")
        cap = cv2.VideoCapture(str(result_path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame, caption="Frame 0", use_container_width=True)
        else:
            st.warning("⚠️ Không đọc được frame đầu tiên!")

        # Nút tải về
        with open(result_path, "rb") as f:
            st.download_button("⬇️ Tải video kết quả", f, file_name=result_path.name, mime="video/mp4")

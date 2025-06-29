# 🚚 Dispatch‑Monitoring‑System

Intelligent monitoring system for a commercial kitchen's dispatch area.

---

## 📹 Demo

- **Video demo model inference:**  https://youtu.be/O5Kx10YaMz0  
- **Video demo display system:** https://youtu.be/_YJ33nJ0lkM

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/HTN-DT-Beo/Dispatch-Monitoring-System.git
cd Dispatch-Monitoring-System
```
### 2. Check Docker
```bash
docker --version          # Check Docker
docker-compose --version  # Check Docker Compose
```

### 3. Containerizing project

Run below command to create and start containers of Dispatch‑Monitoring‑System

```cmd
docker-compose up --build
```

### 4. Access the running services
- 📘 API Docs (FastAPI): http://localhost:8000/docs
- 📊 Dashboard (Streamlit): http://localhost:8501


## 📝 Project Summary

### 📌 Components Implemented

- **Detection:** YOLOv8  
- **Classification:** EfficientNet-B0  
- **Tracking:** BoT-SORT  

---

### ✅ Results

- 📷 **Image Output**  
![Demo Pic](https://github.com/HTN-DT-Beo/Dispatch-Monitoring-System/blob/main/demo.jpg)

---

### ⚠️ Limitations

- Time constraint: Only **3 days** were available for implementation, which limited the system's completeness.
- **Mismatch in training and real-world camera angles**:  
  The model was trained on images from a different angle than the actual camera footage. As a result, performance on full-view test videos may not be optimal.  
  A cropped video with close-up footage of the food counter yielded much better results.
- **UI limitations**:  
  The interface is still basic and not tailored to actual user needs.
- **Missing feedback loop**:  
  The requirement:  
  *"The system should also include functionality to improve model performance based on user feedback"*  
  has not yet been implemented.

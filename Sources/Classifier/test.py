import cv2, numpy as np, onnxruntime as ort
from pathlib import Path

IMG_SIZE = 224
MEAN, STD = np.array([0.5]*3, np.float32), np.array([0.5]*3, np.float32)
LABELS_DISH = ["empty", "kakigori", "not_empty"]

dish_sess = ort.InferenceSession("Models/Classification/dish_cls.onnx",
                            providers=["CPUExecutionProvider"])

tray_sess = ort.InferenceSession("Models/Classification/tray_cls.onnx",
                            providers=["CPUExecutionProvider"])

def classify_img(sess, path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    im  = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    im  = ((im - MEAN) / STD).transpose(2, 0, 1)[None]          # (1,3,H,W)
    logits = sess.run(None, {"images": im})[0]
    return LABELS_DISH[int(np.argmax(logits, 1)[0])]


print(classify_img(dish_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//dish//empty//00000000252000000_frame0_jpg.rf.5c6e9f1a49b46860d430e79b9890813b_3_0.jpg"))
print(classify_img(dish_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//dish//kakigori//00000001066000000_frame68850_jpg.rf.4a13b788ea73e62c806a34049b450775_5_23.jpg"))
print(classify_img(dish_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//dish//not_empty//00000000252000000_frame2100_jpg.rf.a5e543f65624e88cea72f264c6d64632_5_10.jpg"))


print(classify_img(tray_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//tray//empty//00000000252000000_frame750_jpg.rf.e081e0985c375282c477f466c88787d7_4_407.jpg"))
print(classify_img(tray_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//tray//kakigori//00000000252000000_frame68550_jpg.rf.04553ce072a64acb5dae7a72089e2925_8_8.jpg"))
print(classify_img(tray_sess, "D://Project//Dispatch_Monitoring_System//Dispatch-Monitoring-System//Data//Dataset//Classification//tray//not_empty//00000000252000000_frame8700_jpg.rf.2052514c7500b9e3e31880cfe6931515_4_490.jpg"))

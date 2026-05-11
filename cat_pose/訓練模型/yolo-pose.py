from flask import Flask, Response, jsonify
import cv2
from ultralytics import YOLO
import threading
import requests
import time
import socket
import math

# ============================================
# 📡 Node-RED 配置 (請確認 Node-RED 電腦的 IP)
# ============================================
NODE_RED_HOST = "127.0.0.1" # 如果 Node-RED 在同一台電腦，改用 "127.0.0.1"
NODE_RED_PORT = 1880
URL_NOTIFY = f"http://{NODE_RED_HOST}:{NODE_RED_PORT}/python_online"
URL_RESULT = f"http://{NODE_RED_HOST}:{NODE_RED_PORT}/yolo_result"

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

LOCAL_IP = get_ip()

# ============================================
# 🐱 載入 17 點 YOLO Pose 模型
# ============================================
print("🔄 載入 17 點 YOLO Pose 模型...")    
# 請確保此模型輸出為 17 個點 (如 yolov8n-pose.pt)
model = YOLO(r"C:\tinycnn\adam.pt") 
print("✅ 模型載入成功（17 keypoints 版本）")

# 標準 17 點骨架連線 (COCO 格式)
CAT_SKELETON_17 = [
    (0,1),(0,2),(1,3),(2,4),          # 臉部
    (5,6),(5,11),(6,12),(11,12),      # 軀幹 (5:L-Sh, 6:R-Sh, 11:L-Hip, 12:R-Hip)
    (5,7),(7,9),(6,8),(8,10),         # 前肢 (9,10 為前爪)
    (11,13),(13,15),(12,14),(14,16)   # 後肢 (15,16 為後爪)
]

# ============================================
# 🐱 姿態辨識邏輯 (17 點專用)
# ============================================
def classify_posture_17(kp, activity=0):
    # 索引對應：0:鼻, 5:左肩, 6:右肩, 11:左臀, 12:右臀, 9:左前掌, 10:右前掌, 15:左後掌, 16:右後掌
    Nose = kp[0]
    L_Shoulder, R_Shoulder = kp[5], kp[6]
    L_Hip, R_Hip = kp[11], kp[12]
    
    # 計算身體中心點與地面高度
    avg_shoulder_y = (L_Shoulder[1] + R_Shoulder[1]) / 2
    avg_hip_y = (L_Hip[1] + R_Hip[1]) / 2
    
    # 找尋四肢中最低的點作為地面參考
    ground_y = max(kp[9][1], kp[10][1], kp[15][1], kp[16][1])
    
    sh_h = ground_y - avg_shoulder_y
    hip_h = ground_y - avg_hip_y

    # 1. BACK (背對: 鼻子在肩部下方或遮蔽)
    if Nose[1] > avg_shoulder_y and avg_shoulder_y < avg_hip_y:
        return "BACK"

    # 2. SIT (坐: 肩膀高，屁股低)
    if sh_h > 40 and hip_h < 30:
        return "SIT"

    # 3. LAY (躺: 全身高度都很低)
    if sh_h < 25 and hip_h < 25:
        return "LAY"

    return "STAND"

# ============================================
# 🎥 串流處理
# ============================================
cap = cv2.VideoCapture(r"C:\tinycnn\cat5.mp4")
app = Flask(__name__)

def generate_frames():
    global last_send_time
    last_send_time = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 影片重播
            continue

        results = model(frame, conf=0.5, verbose=False)[0]
        posture = "UNKNOWN"
        
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kp = results.keypoints.xy[0].cpu().numpy()
            if len(kp) >= 17:
                posture = classify_posture_17(kp)
                # 畫圖邏輯 (略，與之前相似但連線改用 CAT_SKELETON_17)
                for a, b in CAT_SKELETON_17:
                    pt1 = tuple(map(int, kp[a]))
                    pt2 = tuple(map(int, kp[b]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # 傳送結果給 Node-RED
        if time.time() - last_send_time > 0.5:
            send_result_to_nodered({"posture": posture, "ip": LOCAL_IP})
            last_send_time = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def send_result_to_nodered(data):
    def _post():
        try: requests.post(URL_RESULT, json=data, timeout=0.2)
        except: pass
    threading.Thread(target=_post, daemon=True).start()

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # 強制使用 5000 埠以配合您的 Node-RED 錯誤訊息
    print(f"📺 串流網址: http://{LOCAL_IP}:5000/stream")
    app.run(host="0.0.0.0", port=5000, threaded=True)
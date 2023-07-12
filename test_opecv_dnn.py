import cv2
import numpy as np
import time

# Load model YOLOv4-tiny pre-trained trên COCO dataset
net = cv2.dnn.readNet(r'.\MobileNetV2-YOLOv3-Nano-voc.weights', r'.\MobileNetV2-YOLOv3-Nano-voc.cfg')

# List các tên lớp trong COCO dataset
classes = []
with open(r'.\voc.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Khởi tạo font chữ
font = cv2.FONT_HERSHEY_SIMPLEX

# Màu sắc cho tất cả các đối tượng
color = (0, 255, 0)

# Đọc video từ camera
cap = cv2.VideoCapture(r"rtsp://admin:Atc123456@192.168.1.220:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")

# Khởi tạo biến thời gian để tính fps và độ trễ
prev_time = 0
delay_time = 0

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    
    # Chuyển frame thành blob để đưa vào mạng YOLOv4-tiny
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Lấy thông tin output từ mạng YOLOv4-tiny
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)
    
    # Khởi tạo các list để lưu thông tin về đối tượng
    boxes = []
    confidences = []
    class_ids = []
    color = (0, 255, 0)
    # Xử lý output để lấy thông tin về đối tượng
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Áp dụng non-max suppression để loại bỏ các box trùng nhau
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Vẽ hình chữ nhật và viết chữ trên hình chữ nhật cho các đối tượng còn lại
    # Màu sắc cho tất cả các đối tượng
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f'{label} {confidence:.2f}'
            cv2.putText(frame, text, (x, y - 5), font, 1, color, 2)

    # Tính fps và độ trễ
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    fps = 1 / elapsed_time
    delay_time = max(1, int((1000/fps) - elapsed_time))

    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(frame, fps_text, (10, 50), font, 1, color, 2)

    delay_text = f'Delay: {delay_time} ms'
    cv2.putText(frame, delay_text, (10, 100), font, 1, color, 2)

    cv2.imshow('frame', frame)

    # Thoát khỏi vòng lặp khi nhấn phím ESC
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
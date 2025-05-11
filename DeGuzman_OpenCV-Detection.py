import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Can't open video capture")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps} frames per second")

print(f"Loading object detection model...")
model = YOLO("DeGuzman_YOLOv8n_best.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, conf=0.2)

    if len(results) > 0:
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            class_ids = result.boxes.cls

            for box, conf, class_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(class_id)]
                confidence = float(conf)

                # draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # display class name on top-left
                class_name_text = f"{label}"
                cv2.putText(frame, class_name_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                # show confidence level on top-right
                conf_level_text = f"{confidence:.2f}%"
                conf_x = x2 - cv2.getTextSize(conf_level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                cv2.putText(frame, conf_level_text, (conf_x, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(f"Display", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

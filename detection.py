import shutil
import tempfile
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

TARGET_CLASSES = {
  0: "person",
  39: "bottle",
  56: "chair",
  62: "tv",
  63: "laptop",
  64: "mouse",
  66: "keyboard",
  67: "cell phone",
  73: "book",
}

AGE_BUCKETS = [
  "(0-2)",
  "(4-6)",
  "(8-12)",
  "(15-20)",
  "(25-32)",
  "(38-43)",
  "(48-53)",
  "(60-100)",
]

FACE_COLOR = (255, 180, 0)
OBJECT_COLORS = [
  (0, 255, 0),
  (0, 200, 255),
  (255, 0, 255),
  (255, 255, 0),
  (0, 128, 255),
]

MODEL_DIR = Path(__file__).parent / "models"
AGE_PROTO = MODEL_DIR / "age_deploy.prototxt"
AGE_MODEL = MODEL_DIR / "age_net.caffemodel"

MODEL_URLS = {
  AGE_PROTO: "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
  AGE_MODEL: "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/5b60d9f8a8608cdbbcdaaa39bf28f351e8d8553b/model/age_net.caffemodel",
}


def download_file(url, path):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists():
    urllib.request.urlretrieve(url, path)


def load_face_detector():
  cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
  detector = cv2.CascadeClassifier(str(cascade_path))
  if detector.empty():
    raise RuntimeError("Failed to load face detector")
  return detector


def load_age_model():
  for path, url in MODEL_URLS.items():
    download_file(url, path)

  tmp_dir = Path(tempfile.gettempdir()) / "ai_dz2_models"
  tmp_dir.mkdir(parents=True, exist_ok=True)
  tmp_proto = tmp_dir / "age_deploy.prototxt"
  tmp_model = tmp_dir / "age_net.caffemodel"
  shutil.copy2(AGE_PROTO, tmp_proto)
  shutil.copy2(AGE_MODEL, tmp_model)

  net = cv2.dnn.readNetFromCaffe(str(tmp_proto), str(tmp_model))
  return net


def detect_faces(frame, detector):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
  return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def estimate_age(frame, face_rect, age_net):
  x, y, w, h = face_rect
  h_frame, w_frame = frame.shape[:2]

  pad = int(0.1 * w)
  x1 = max(0, x - pad)
  y1 = max(0, y - pad)
  x2 = min(w_frame, x + w + pad)
  y2 = min(h_frame, y + h + pad)

  face = frame[y1:y2, x1:x2]
  if face.size == 0:
    return "?"

  blob = cv2.dnn.blobFromImage(
    face,
    1.0,
    (227, 227),
    (78.4263377603, 87.7689143744, 114.895847746),
    swapRB=False,
  )
  age_net.setInput(blob)
  preds = age_net.forward()
  age_idx = int(preds[0].argmax())
  return AGE_BUCKETS[age_idx]


def draw_faces(frame, faces, age_net):
  for face_rect in faces:
    x, y, w, h = face_rect
    age = estimate_age(frame, face_rect, age_net)

    cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_COLOR, 2)
    cv2.putText(
      frame,
      f"face {age}",
      (x, y - 8),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      FACE_COLOR,
      2,
    )


def detect_objects(model, frame, conf=0.35, imgsz=640):
  results = model.predict(
    frame,
    classes=list(TARGET_CLASSES.keys()),
    conf=conf,
    imgsz=imgsz,
    verbose=False,
  )
  return results[0]


def draw_objects(frame, result):
  if result.boxes is None:
    return

  for box in result.boxes:
    class_id = int(box.cls[0])
    label = TARGET_CLASSES.get(class_id, "object")
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = OBJECT_COLORS[class_id % len(OBJECT_COLORS)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def run_camera(camera_id=0):
  cap = cv2.VideoCapture(camera_id)
  if not cap.isOpened():
    raise RuntimeError(f"Camera {camera_id} not available")

  model = YOLO("yolov8n.pt")
  face_detector = load_face_detector()
  age_net = load_age_model()

  while True:
    ok, frame = cap.read()
    if not ok:
      break

    faces = detect_faces(frame, face_detector)
    draw_faces(frame, faces, age_net)

    result = detect_objects(model, frame)
    draw_objects(frame, result)

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  run_camera()

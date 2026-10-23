
from pathlib import Path

from ultralytics import YOLO


class TomatoDetector:
    """Detect individual tomatoes in an image using YOLO."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image_path, confidence_threshold=0.40):
        """Return tomato bounding boxes and detection confidence."""

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for box, confidence, class_id in zip(
                boxes, confidences, class_ids
            ):
                class_id = int(class_id)
                class_name = result.names[class_id]

                if class_name.lower() not in {"tomato", "tomatoes"}:
                    continue

                x1, y1, x2, y2 = map(int, box)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "detection_confidence": float(confidence),
                    "class_name": class_name
                })

        return detections


import cv2
from PIL import Image
import numpy as np
import argparse
from omegaconf import OmegaConf

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


def main(config):
    face_recognizer = cv2.CascadeClassifier(config.model.face_detector.name_or_path)

    processor = ViTImageProcessor.from_pretrained(config.model.emotion_classifier.name_or_path)
    model = ViTForImageClassification.from_pretrained(config.model.emotion_classifier.name_or_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read() # Capture frame-by-frame
        
        if not ret:
            print("Error: failed to capture image")
            break

        face = face_recognizer.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face) > 0:
            face = sorted(face, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            roi = frame[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (config.crop_size, config.crop_size))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(roi)
            inputs = processor(images=image, return_tensors="pt")

            outputs = model(**inputs)
            logits = outputs.logits

            predicted_class_idx = logits.argmax(-1).item()

            prediction = model.config.id2label[predicted_class_idx]
            print(f"Predicted class: {prediction}")

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=frame, text=prediction, org=(50, 50), fontFace=font, fontScale=1, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_4)

            cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="inference_config", help="config file path")
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    main(config)
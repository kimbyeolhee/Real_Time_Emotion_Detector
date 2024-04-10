from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import cv2


model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: failed to capture image")
        break

    image = Image.fromarray(frame)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item() # imagenet 1000 class로 학습해놓은 거라 1000개의 class 중 하나가 나옴

    prediction = model.config.id2label[predicted_class_idx]
    print(f"Predicted class: {prediction}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img=frame, text=prediction, org=(50, 50), fontFace=font, fontScale=1, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_4)

    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
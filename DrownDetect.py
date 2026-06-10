import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from PIL import Image
import time




lb = joblib.load('lb.pkl')
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # changed 3 to 1
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(lb.classes_))
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print('Loading model and label binarizer...')
lb = joblib.load('lb.pkl')
model = CustomCNN()
print('Model Loaded...')
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
print('Loaded model state_dict...')
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

t0 = time.time() #gives time in seconds after 1970

def detectDrowning(source):

    drowning_frames = 0
    normal_frames = 0

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return {
            "is_drowning": False,
            "drowning_frames": 0,
            "normal_frames": 0,
            "risk_score": 0,
            "error": "Could not open video"
        }

    fram = 0

    while cap.isOpened():

        status, frame = cap.read()

        if not status or frame is None:
            break

        fram += 1

        # Analyze only every 5th frame
        if fram % 5 != 0:
            continue

        # Stop after 200 frames
        if fram >= 200:
            break

        try:
            bbox, label, conf = cv.detect_common_objects(frame)
        except Exception:
            continue

        # Single person detected
        if len(bbox) == 1:

            with torch.no_grad():

                pil_image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                pil_image = aug(
                    image=np.array(pil_image)
                )["image"]

                pil_image = np.transpose(
                    pil_image,
                    (2, 0, 1)
                ).astype(np.float32)

                pil_image = torch.tensor(
                    pil_image,
                    dtype=torch.float
                ).unsqueeze(0)

                outputs = model(pil_image)

                _, preds = torch.max(outputs.data, 1)

                prediction = lb.classes_[preds.item()]

                if prediction == "drowning":
                    drowning_frames += 1
                else:
                    normal_frames += 1

        # Multiple people detected
        elif len(bbox) > 1:

            centres = []

            for bbox_i in bbox:
                centre_i = [
                    (bbox_i[0] + bbox_i[2]) / 2,
                    (bbox_i[1] + bbox_i[3]) / 2
                ]
                centres.append(centre_i)

            distances = []

            for i in range(len(centres)):
                for j in range(i + 1, len(centres)):

                    dist = np.sqrt(
                        (centres[i][0] - centres[j][0]) ** 2 +
                        (centres[i][1] - centres[j][1]) ** 2
                    )

                    distances.append(dist)

            if len(distances) > 0 and min(distances) < 50:
                drowning_frames += 1
            else:
                normal_frames += 1

    cap.release()

    total_frames = drowning_frames + normal_frames

    if total_frames == 0:
        risk_score = 0
    else:
        risk_score = (
            drowning_frames / total_frames
        ) * 100

    return {
        "is_drowning": risk_score >= 70,
        "drowning_frames": drowning_frames,
        "normal_frames": normal_frames,
        "risk_score": risk_score
    }
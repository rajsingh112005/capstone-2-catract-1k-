"""Configuration settings for the Surgical Safety Agent."""

import os
import torch
from torchvision import transforms

os.environ["QT_QPA_PLATFORM"] = "xcb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model_fold_5.pth"
NUM_CLASSES = 13
AI_PROCESS_INTERVAL = 0.1

SURGICAL_TIMELINE = [
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation_Aspiration",
    "Capsule_Polishing",
    "Lens_Implantation",
    "Lens_Positioning",
    "Tonifying_Antibiotics",
    "Viscoelastic_Suction",
    "Anterior_Chamber_Flushing",
    "Idle",
]

PHASE_LIMITS = {
    "Incision": 45,
    "Viscoelastic": 60,
    "Phacoemulsification": 300, # 5 minutes
    "Capsulorhexis": 120
}

TRANSFORM_PIPELINE = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

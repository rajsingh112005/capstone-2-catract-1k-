import cv2
from PIL import Image
from surgical_agent.config import TRANSFORM_PIPELINE


def preprocess_frame(frame):
    """
    Preprocess a video frame for model inference.

    Args:
        frame (numpy.ndarray): Input frame in BGR format from OpenCV.

    Returns:
        torch.Tensor: Transformed tensor ready for model input.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = TRANSFORM_PIPELINE(img_pil)
    return img_tensor

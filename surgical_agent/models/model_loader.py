
import torch
from surgical_agent.config import DEVICE, MODEL_PATH
from .surgical_vit_agent import SurgicalViTAgent


def load_model(model_path=None, num_classes=13, device=None):
    """
    Load and initialize the Surgical ViT Agent model.

    Args:
        model_path (str, optional): Path to the model checkpoint. Defaults to MODEL_PATH.
        num_classes (int, optional): Number of surgical phases. Defaults to 13.
        device (torch.device, optional): Device to load model on. Defaults to DEVICE.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if model_path is None:
        model_path = MODEL_PATH
    if device is None:
        device = DEVICE

    model = SurgicalViTAgent(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

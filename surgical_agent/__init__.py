"""Autonomous Surgical Safety Agent - Main Package."""

__version__ = "1.0.0"
__author__ = "ML GPU Capstone Team"

from surgical_agent.config import (
    DEVICE,
    MODEL_PATH,
    NUM_CLASSES,
    AI_PROCESS_INTERVAL,
    SURGICAL_TIMELINE,
    TRANSFORM_PIPELINE,
)
from surgical_agent.models import SurgicalViTAgent, load_model
from surgical_agent.utils import LiveCameraSimulator, preprocess_frame
from surgical_agent.ui import render_overlay

__all__ = [
    "DEVICE",
    "MODEL_PATH",
    "NUM_CLASSES",
    "AI_PROCESS_INTERVAL",
    "SURGICAL_TIMELINE",
    "TRANSFORM_PIPELINE",
    "SurgicalViTAgent",
    "load_model",
    "LiveCameraSimulator",
    "preprocess_frame",
    "render_overlay",
]

#!/usr/bin/env python3
import collections
import time

import cv2
import torch

from surgical_agent.config import (
    DEVICE,
    AI_PROCESS_INTERVAL,
    SURGICAL_TIMELINE,
)
from surgical_agent.models import load_model
from surgical_agent.utils import LiveCameraSimulator, preprocess_frame
from surgical_agent.ui import render_overlay


def main():
    """Main application loop for the Surgical Safety Agent."""
    print(f"--- AGENT DEPLOYING ON {DEVICE} ---")
    model = load_model()
    print("✓ Model loaded successfully")

    vs = LiveCameraSimulator("case_2001.mp4").start()
    print(f"✓ Video stream initialized at {vs.fps} FPS")

    frame_buffer = collections.deque(maxlen=16)

    current_phase = "Detecting..."
    confidence_val = 0.0
    last_ai_time = 0

    print("\n--- SURGICAL SAFETY AGENT ACTIVE ---\n")

    try:
        while True:
            frame_start = time.time()

            frame = vs.read()
            if frame is None:
                print("End of video stream reached.")
                break

            img_tensor = preprocess_frame(frame)
            frame_buffer.append(img_tensor)

            now = time.time()
            if len(frame_buffer) == 16 and (now - last_ai_time) > AI_PROCESS_INTERVAL:
                input_seq = (
                    torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
                )
                with torch.no_grad():
                    output = model(input_seq)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred_idx = torch.max(prob, 1)
                    current_phase = SURGICAL_TIMELINE[pred_idx.item()]
                    confidence_val = conf.item()
                last_ai_time = now

      
            display_frame = render_overlay(frame, current_phase, confidence_val)

            cv2.imshow(
                "Surgical Agent - 1x Real-Time Simulation", display_frame
            )

            elapsed = time.time() - frame_start
            sleep_time = vs.frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

   
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nAgent shutdown requested by user.")
                break

    except KeyboardInterrupt:
        print("\nAgent interrupted by user (Ctrl+C).")
    finally:
        vs.stop()
        cv2.destroyAllWindows()
        print("✓ Resources released successfully")
        print("\n--- AGENT SHUTDOWN COMPLETE ---\n")


if __name__ == "__main__":
    main()

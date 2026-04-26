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
from surgical_agent.reasoning import SurgicalReasoningAgent
from surgical_agent.logger import SurgicalLogger


def main():
    """Main application loop for the Surgical Safety Agent."""
    print(f"--- AGENT DEPLOYING ON {DEVICE} ---")
    model = load_model()
    print("✓ Model loaded successfully")

    vs = LiveCameraSimulator("case_2001.mp4").start()
    print(f"✓ Video stream initialized at {vs.fps} FPS")

    reasoner = SurgicalReasoningAgent(SURGICAL_TIMELINE)
    logger = SurgicalLogger()
    frame_buffer = collections.deque(maxlen=16)

    frame_idx = 0

    active_phase = "Initializing..."
    active_alerts = []
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

            frame_idx += 1
            video_time_sec = frame_idx / vs.fps

            now = time.time()
            if len(frame_buffer) == 16 and (now - last_ai_time) > AI_PROCESS_INTERVAL:
                input_seq = (
                    torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
                )
                with torch.no_grad():
                    output = model(input_seq)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred_idx = torch.max(prob, 1)
                    
                    raw_phase = SURGICAL_TIMELINE[pred_idx.item()]
                    confidence_val = conf.item()

                    # Pass raw prediction to the reasoning layer
                    active_phase, active_alerts = reasoner.verify_step(raw_phase, confidence_val)

                    # Save to log
                    logger.add_entry(frame_idx, video_time_sec, active_phase, confidence_val, active_alerts)
                
                last_ai_time = now

      
            display_frame = render_overlay(frame, active_phase, confidence_val, active_alerts, frame_idx, video_time_sec)

            cv2.imshow(
                "Surgical Safety Agent (Idea 1 + 3)", display_frame
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
        logger.generate_final_report()
        print("✓ Resources released successfully")
        print("\n--- AGENT SHUTDOWN COMPLETE ---\n")


if __name__ == "__main__":
    main()

import cv2


def render_overlay(frame, current_phase, confidence_val, alerts, frame_idx=0, video_time_sec=0.0):
    """
    Render the UI overlay on the video frame.

    Args:
        frame (numpy.ndarray): Input frame to render on.
        current_phase (str): Current surgical phase being detected.
        confidence_val (float): Confidence score (0-1) for the prediction.
        alerts (list): List of anomaly alert strings.
        frame_idx (int): Current frame index.
        video_time_sec (float): Current video time in seconds.

    Returns:
        numpy.ndarray: Frame with UI overlay applied.
    """
    display_frame = frame.copy()

    # Header bar
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 100), (0, 0, 0), -1)
    
    # Current Phase
    cv2.putText(display_frame, f"PHASE: {current_phase}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)
    
    # AI Confidence
    cv2.putText(display_frame, f"CONF: {confidence_val*100:.1f}%", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Frame and Timestamp
    cv2.putText(display_frame, f"FRAME: {frame_idx} | T: {round(video_time_sec, 1)}s", (200, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Anomaly Alerts (Red)
    for i, alert in enumerate(alerts):
        # Position the alerts right under the header bar to avoid overlapping the green text horizontally
        cv2.putText(display_frame, alert, (20, 140 + (i*35)), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)


    return display_frame

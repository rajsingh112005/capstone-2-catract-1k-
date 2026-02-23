import cv2


def render_overlay(frame, current_phase, confidence_val):
    """
    Render the UI overlay on the video frame.

    Args:
        frame (numpy.ndarray): Input frame to render on.
        current_phase (str): Current surgical phase being detected.
        confidence_val (float): Confidence score (0-1) for the prediction.

    Returns:
        numpy.ndarray: Frame with UI overlay applied.
    """
    display_frame = frame.copy()

    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)

    cv2.putText(
        display_frame,
        f"PHASE: {current_phase}",
        (20, 35),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        display_frame,
        f"CONFIDENCE: {confidence_val*100:.1f}%",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    cv2.circle(
        display_frame, (display_frame.shape[1] - 150, 35), 8, (0, 0, 255), -1
    )
    cv2.putText(
        display_frame,
        "LIVE SIM",
        (display_frame.shape[1] - 130, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    return display_frame

import time

# Max expected duration (seconds) per phase for Anomaly Detection
PHASE_LIMITS = {
    "Incision": 45,
    "Viscoelastic": 60,
    "Phacoemulsification": 300, # 5 minutes
    "Capsulorhexis": 120
}

class SurgicalReasoningAgent:
    def __init__(self, timeline):
        self.timeline = timeline
        self.last_valid_idx = -1
        self.current_phase = "Initializing..."
        self.phase_start_time = time.time()
        self.alerts = []

    def verify_step(self, predicted_phase, confidence):
        now = time.time()
        new_alerts = []
        
        if predicted_phase not in self.timeline or predicted_phase == 'Idle':
            return self.current_phase, []

        pred_idx = self.timeline.index(predicted_phase)

        # 1. SEQUENCE ANOMALY (Idea 3)
        # If jumping more than 1 step ahead, it's a deviation
        if pred_idx > self.last_valid_idx + 1:
            new_alerts.append(f"CRITICAL: Phase Skip ({self.current_phase} -> {predicted_phase})")
        
        # 2. DURATION ANOMALY (Idea 3)
        if predicted_phase == self.current_phase:
            duration = now - self.phase_start_time
            limit = PHASE_LIMITS.get(predicted_phase, 180) # Default 3 mins
            if duration > limit:
                new_alerts.append(f"WARNING: Prolonged {predicted_phase} ({int(duration)}s)")
        else:
            # Phase has changed, update tracking
            self.current_phase = predicted_phase
            self.last_valid_idx = max(self.last_valid_idx, pred_idx)
            self.phase_start_time = now

        # 3. UNCERTAINTY ANOMALY
        if confidence < 0.75:
            new_alerts.append("VISUAL: Low Model Confidence")

        return self.current_phase, new_alerts

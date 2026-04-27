import collections
import time
from surgical_agent.config import AI_PROCESS_INTERVAL

class SurgicalLogger:
    def __init__(self):
        self.logs = []
        self.start_wall_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def add_entry(self, frame_idx, video_time, phase, confidence, alerts):
        # LOGS EVERY AI DECISION POINT
        entry = {
            "timestamp": round(video_time, 2),
            "frame": frame_idx,
            "phase": phase,
            "confidence": round(confidence * 100, 2),
            "alerts": ", ".join(alerts) if alerts else "None"
        }
        self.logs.append(entry)

    def generate_final_report(self):
        print("\n📄 GENERATING AI SURGICAL REPORT...")
        report_path = "surgical_report.txt"
        
        # BASIC REASONING LOGIC (AGENTIC LAYER)
        total_duration = self.logs[-1]['timestamp'] if self.logs else 0
        
        phase_durations = collections.defaultdict(float)
        if self.logs:
            phase_durations[self.logs[0]['phase']] += self.logs[0]['timestamp']
            for i in range(1, len(self.logs)):
                duration = self.logs[i]['timestamp'] - self.logs[i-1]['timestamp']
                phase_durations[self.logs[i]['phase']] += duration

        anomaly_logs = [log for log in self.logs if log['alerts'] != "None"]

        with open(report_path, "w") as f:
            f.write(f"=== AUTONOMOUS SURGICAL REPORT ===\n")
            f.write(f"Date: {self.start_wall_time}\n")
            f.write(f"Total Video Duration: {total_duration} seconds\n")
            f.write(f"Status: {'COMPLETED WITH ANOMALIES' if anomaly_logs else 'SUCCESSFUL'}\n\n")
            
            f.write("--- PHASE SUMMARY ---\n")
            for phase, duration in phase_durations.items():
                f.write(f"- {phase}: ~{round(duration, 1)}s\n")
            
            f.write("\n--- AI REASONING OVER ANOMALIES ---\n")
            if not anomaly_logs:
                f.write("No procedural deviations detected. Surgeon followed Golden Path.\n")
            else:
                for anomaly in anomaly_logs:
                    f.write(f"[{anomaly['timestamp']}s] {anomaly['alerts']}\n")
                    # AGENTIC INSIGHT
                    if "Phase Skip" in anomaly['alerts']:
                        f.write("   > INSIGHT: Potential risk of tissue trauma due to bypassed preparation step.\n")
                    if "Prolonged" in anomaly['alerts']:
                        f.write("   > INSIGHT: Suggests mechanical resistance or visibility issues during this phase.\n")

        print(f"✅ Report saved to {report_path}")

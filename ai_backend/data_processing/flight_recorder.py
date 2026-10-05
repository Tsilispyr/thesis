"""Records raw UDP telemetry packets from the Godot simulation to CSV.

Used to build new, motor/barometer/magnetometer-inclusive training datasets
from live flights, now that drone_body.gd's telemetry is physically real
(see AI_RECOVERY_EXECUTION_PLAN.md, Phase 1 Task 1/3). The existing offline
CSVs (uav_navigation_dataset.csv, imu_data.csv) have no motor-thrust column
at all, so this is how that data comes to exist.

One row per received packet, one file per recording session. Motor_out
(a 4-element list in the payload) is flattened into motor_0..motor_3 columns
so the file loads directly as a flat table in pandas.
"""
import csv
import os
import time

FIELDNAMES = [
    "recv_time", "drone_id", "timestamp", "gps_signal",
    "lat", "lon", "alt",
    "imu_acc_x", "imu_acc_y", "imu_acc_z",
    "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
    "roll", "pitch", "yaw",
    "mag_x", "mag_y", "mag_z",
    "baro_alt",
    "motor_0", "motor_1", "motor_2", "motor_3",
    "speed", "battery",
]


class FlightRecorder:
    """Appends one CSV row per telemetry packet handed to record()."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        fname = time.strftime("flight_%Y%m%d_%H%M%S.csv")
        self.path = os.path.join(out_dir, fname)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._row_count = 0
        print(f"[FlightRecorder] Recording telemetry to {self.path}")

    def record(self, data: dict) -> None:
        motor_out = data.get("motor_out", [0.0, 0.0, 0.0, 0.0])
        row = {
            "recv_time":  time.time(),
            "drone_id":   data.get("drone_id", ""),
            "timestamp":  data.get("timestamp", ""),
            "gps_signal": data.get("gps_signal", True),
            "lat": data.get("lat", 0.0), "lon": data.get("lon", 0.0), "alt": data.get("alt", 0.0),
            "imu_acc_x":  data.get("imu_acc_x", 0.0),
            "imu_acc_y":  data.get("imu_acc_y", 0.0),
            "imu_acc_z":  data.get("imu_acc_z", 0.0),
            "imu_gyro_x": data.get("imu_gyro_x", 0.0),
            "imu_gyro_y": data.get("imu_gyro_y", 0.0),
            "imu_gyro_z": data.get("imu_gyro_z", 0.0),
            "roll":  data.get("roll", 0.0),
            "pitch": data.get("pitch", 0.0),
            "yaw":   data.get("yaw", 0.0),
            "mag_x": data.get("mag_x", 0.0), "mag_y": data.get("mag_y", 0.0), "mag_z": data.get("mag_z", 0.0),
            "baro_alt": data.get("baro_alt", 0.0),
            "motor_0": motor_out[0] if len(motor_out) > 0 else 0.0,
            "motor_1": motor_out[1] if len(motor_out) > 1 else 0.0,
            "motor_2": motor_out[2] if len(motor_out) > 2 else 0.0,
            "motor_3": motor_out[3] if len(motor_out) > 3 else 0.0,
            "speed":   data.get("speed", 0.0),
            "battery": data.get("battery", 0.0),
        }
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % 500 == 0:
            self._fh.flush()

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()
        print(f"[FlightRecorder] Saved {self._row_count} rows → {self.path}")

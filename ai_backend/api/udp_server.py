import socket
import json
import threading
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.flight_recorder import FlightRecorder
from api.recovery_orchestrator import RecoveryOrchestrator, MODEL_REGISTRY


class UDPServer:
    def __init__(self, listen_ip="0.0.0.0", listen_port=14551,
                 godot_ip="127.0.0.1", godot_port=14552, record_dir=None,
                 model_name="LSTM", use_rl_seek=False):
        self.listen_ip   = listen_ip
        self.listen_port = listen_port
        self.godot_ip    = godot_ip
        self.godot_port  = godot_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.listen_ip, self.listen_port))

        self.running = False
        self._orchestrator = RecoveryOrchestrator(model_name=model_name, use_rl_seek=use_rl_seek)
        self._recorder = FlightRecorder(record_dir) if record_dir else None

        print(f"UDP Server listening on {self.listen_ip}:{self.listen_port}")
        print(f"UDP Server will send commands to {self.godot_ip}:{self.godot_port}")

    def start(self):
        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.sock.close()
        if self._recorder is not None:
            self._recorder.close()

    def _listen_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
                if data:
                    try:
                        self.handle_telemetry(json.loads(data.decode('utf-8')))
                    except json.JSONDecodeError:
                        print("Invalid JSON received")
            except Exception as e:
                if self.running:
                    print(f"UDP Receive Error: {e}")

    def handle_telemetry(self, data: dict):
        drone_id = data.get("drone_id", "Unknown")
        has_gps  = data.get("gps_signal", True)

        if self._recorder is not None:
            self._recorder.record(data)

        # decide() is called on every packet regardless of GPS state -- it needs
        # a continuously-filling feature buffer and an up-to-date "last known
        # good position" anchor ready for the instant GPS actually drops. Only
        # the RESULT is worth transmitting back once GPS is genuinely lost.
        result = self._orchestrator.decide(
            drone_id, data, forced_layer=data.get("force_layer"))

        if not has_gps:
            self.send_command({"drone_id": drone_id,
                               "action": "ai_correction",
                               "layer": result["layer"],
                               "goal": result["goal"],
                               "velocity_cmd": result["velocity_cmd"]})
            print(f"[{drone_id}] GPS LOST -> layer={result['layer']}  goal={result['goal']}  "
                  f"v={result['velocity_cmd']}")

    def send_command(self, payload_dict: dict):
        try:
            self.sock.sendto(json.dumps(payload_dict).encode('utf-8'),
                             (self.godot_ip, self.godot_port))
        except Exception as e:
            print(f"Error sending UDP command: {e}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true",
                     help="Record all received telemetry to a new CSV under "
                          "datasets/recorded/ (one file per session, timestamped).")
    ap.add_argument("--record-dir", type=str, default=None,
                     help="Override the recording output directory "
                          "(default: <AI_Recovery>/datasets/recorded/).")
    ap.add_argument("--model", type=str, default="LSTM", choices=list(MODEL_REGISTRY),
                     help="Which trained architecture backs the LSTM fail-safe layer "
                          "(default: LSTM). The reported layer name stays 'LSTM' either "
                          "way -- this only picks which checkpoint answers for it.")
    ap.add_argument("--rl-seek", action="store_true",
                     help="Use the learned RL seek policy instead of the hand-coded "
                          "formula for the seek-bias term (default: off -- opt-in, same "
                          "reasoning as RecoveryOrchestrator's own use_rl_seek default). "
                          "Degrades cleanly to the hand-coded formula if no checkpoint "
                          "exists at ai_backend/models/ppo_recovery_seek.zip.")
    args = ap.parse_args()

    record_dir = None
    if args.record or args.record_dir:
        record_dir = args.record_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets", "recorded")

    server = UDPServer(record_dir=record_dir, model_name=args.model, use_rl_seek=args.rl_seek)
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping UDP Server")
        server.stop()

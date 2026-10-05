FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# UDP ports: 14551 = incoming telemetry from Godot
#            14552 = outgoing AI corrections to Godot
EXPOSE 14551/udp
EXPOSE 14552/udp

CMD ["python", "api/udp_server.py"]

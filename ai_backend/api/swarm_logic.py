import math
from api.path_recovery import calculate_distance

class SwarmInterlink:
    def __init__(self, comm_range=500.0):
        # Maximum communication range in meters
        self.comm_range = comm_range
        
        # Active drone states {drone_id: {'lat': lat, 'lon': lon, 'has_signal': True/False}}
        self.drones = {}
        
    def update_drone_state(self, drone_id, lat, lon, has_signal):
        self.drones[drone_id] = {'lat': lat, 'lon': lon, 'has_signal': has_signal}
        
    def get_relay_drone(self, lost_drone_id):
        """
        If a drone loses signal, find the closest drone that HAS signal
        and is within communication range to act as a relay.

        Always returns a (relay_id, distance) 2-tuple, never a bare None --
        relay_id is None and distance is float('inf') when no relay is
        available (unknown drone id, or no in-range drone has signal),
        so every caller can unpack the result the same way without a
        TypeError on the "no relay" case.
        """
        if lost_drone_id not in self.drones:
            return None, float('inf')

        lost_drone = self.drones[lost_drone_id]

        min_dist = float('inf')
        best_relay = None

        for d_id, state in self.drones.items():
            if d_id == lost_drone_id:
                continue

            if state['has_signal']:
                dist = calculate_distance(lost_drone['lat'], lost_drone['lon'], state['lat'], state['lon'])
                if dist <= self.comm_range and dist < min_dist:
                    min_dist = dist
                    best_relay = d_id

        return best_relay, min_dist

# Test
if __name__ == "__main__":
    swarm = SwarmInterlink(comm_range=1000.0)
    swarm.update_drone_state("Drone_A", 37.7800, -122.4200, False) # Lost signal
    swarm.update_drone_state("Drone_B", 37.7850, -122.4250, True)  # Has signal
    swarm.update_drone_state("Drone_C", 37.7900, -122.4300, True)  # Has signal, further away
    
    relay, dist = swarm.get_relay_drone("Drone_A")
    print(f"Drone_A can interlink with relay {relay} at distance {dist:.2f}m")

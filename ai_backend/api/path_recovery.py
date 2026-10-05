import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r

class PathRecovery:
    def __init__(self):
        # List of known safe zones / signal pockets: (lat, lon, alt)
        self.known_signal_zones = [
            (37.77239080237695, -122.42152728959445, 200.0),
            (37.7839142861282, -122.4199312868125, 150.0)
        ]
        
    def find_nearest_signal_zone(self, current_lat, current_lon):
        """
        Finds the nearest known signal zone.
        Returns coordinates of the zone.
        """
        min_dist = float('inf')
        nearest_zone = None
        
        for zone in self.known_signal_zones:
            dist = calculate_distance(current_lat, current_lon, zone[0], zone[1])
            if dist < min_dist:
                min_dist = dist
                nearest_zone = zone
                
        return nearest_zone, min_dist
        
    def calculate_recovery_vector(self, current_lat, current_lon, target_zone):
        """
        Calculates a simple vector pointing towards the target zone.
        """
        if not target_zone:
            return 0, 0
            
        target_lat, target_lon, target_alt = target_zone
        
        # Simple proportional vector
        d_lat = target_lat - current_lat
        d_lon = target_lon - current_lon
        
        # Normalize
        length = math.sqrt(d_lat**2 + d_lon**2)
        if length == 0:
            return 0, 0
            
        return d_lat / length, d_lon / length

# Test
if __name__ == "__main__":
    pr = PathRecovery()
    zone, dist = pr.find_nearest_signal_zone(37.7800, -122.4200)
    print(f"Nearest zone at {dist} meters: {zone}")
    v_lat, v_lon = pr.calculate_recovery_vector(37.7800, -122.4200, zone)
    print(f"Recovery Vector: {v_lat}, {v_lon}")

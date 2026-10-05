import os
import urllib.request
import math

def download_osm_heightmap(lat, lon, size_km, output_path):
    """
    Downloads an approximate heightmap or OSM terrain data for the specified region.
    (This is a placeholder for actual OSM/SRTM fetching logic).
    """
    print(f"Fetching OSM Terrain for Lat: {lat}, Lon: {lon}, Size: {size_km}km")
    
    # In a real implementation, we would use the Overpass API or an SRTM endpoint
    # like open-elevation to generate a GeoTIFF or PNG heightmap.
    
    # For now, generate a synthetic heightmap PNG that Godot can use as a HeightMapShape3D
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed. Cannot generate synthetic heightmap.")
        return
        
    width = 256
    height = 256
    img = Image.new('L', (width, height), color='black')
    
    # Draw some random "mountains"
    draw = ImageDraw.Draw(img)
    import random
    random.seed(42)
    for _ in range(5):
        cx = random.randint(0, width)
        cy = random.randint(0, height)
        radius = random.randint(30, 80)
        # Create a radial gradient for a mountain
        for r in range(radius, 0, -1):
            color = int(255 * (1 - r/radius))
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, fill=color)
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"Saved synthetic OSM heightmap to {output_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, 'assets', 'terrain', 'mountain_heightmap.png')
    download_osm_heightmap(37.7, -122.4, 5.0, out_path)

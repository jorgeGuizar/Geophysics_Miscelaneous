import numpy as np
import shapely 
from shapely.geometry import Polygon, Point 
import matplotlib.pyplot as plt
import time
import json 

start_time = time.time()

def read_xyz_file(filename):
    """Read XYZ file and return coordinates as numpy array"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:  # Ensure we have at least X,Y,Z coordinates
                try:
                    x, y, z = map(float, parts[:3])
                    data.append([x, y, z])
                except ValueError:
                    continue
    return np.array(data)


def read_polygon_file(filename):
    """Read polygon vertices from file"""
    vertices = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:  # Need at least X,Y coordinates
                try:
                    x, y = map(float, parts[:2])
                    vertices.append((x, y))
                except ValueError:
                    continue
    return vertices

def read_polygon_file2(filename, format_type='auto'):
    """
    Read polygon vertices from file with flexible format handling
    Supported format_types: 'auto', 'xy_list', 'esri', 'geojson'
    """
    vertices = []
    
    with open(filename, 'r') as f:
        content = f.read().strip()
        
        # Auto-detect format if requested
        if format_type == 'auto':
            if content.startswith('{') and '"type": "Polygon"' in content:
                format_type = 'geojson'
            elif 'END' in content or content.split('\n')[0].isalpha():
                format_type = 'esri'
            else:
                format_type = 'xy_list'
        
        # Process based on format type
        if format_type == 'xy_list':
            for line in content.split('\n'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        x, y = map(float, parts[:2])
                        vertices.append((x, y))
                    except ValueError:
                        continue
        
        elif format_type == 'esri':
            lines = content.split('\n')
            # Skip the first line (polygon name) and last line (END)
            for line in lines[1:-1]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        x, y = map(float, parts[:2])
                        vertices.append((x, y))
                    except ValueError:
                        continue
        
        elif format_type == 'geojson':
            try:
                geojson = json.loads(content)
                if geojson['type'] == 'Polygon':
                    # GeoJSON has nested arrays (for potential holes)
                    vertices = [(x, y) for x, y in geojson['coordinates'][0]]
            except (json.JSONDecodeError, KeyError):
                pass
    
    # Ensure polygon is closed (first and last points are the same)
    if len(vertices) > 0 and vertices[0] != vertices[-1]:
        vertices.append(vertices[0])
    
    return vertices

def load_geojson_polygon(filename):
    """Load a polygon from a GeoJSON file."""
    with open(filename, 'r') as f:
        geojson = json.load(f)
    
    # Extract coordinates from the GeoJSON
    # Note: Your file shows a LineString, but you mentioned polygon
    # If it's actually a LineString that forms a closed polygon, we'll use that
    features = geojson.get('features', [])
    if features:
        geometry = features[0]['geometry']
        if geometry['type'] == 'Polygon':
            coords = geometry['coordinates'][0]  # Polygon coordinates
        elif geometry['type'] == 'LineString':
            coords = geometry['coordinates']  # LineString coordinates
            # Check if it's closed (first and last points are the same)
            if coords[0] != coords[-1]:
                coords.append(coords[0])  # Close the polygon
        else:
            raise ValueError("Unsupported geometry type in GeoJSON")
    else:
        # Handle case where features array is empty (direct geometry)
        geometry = geojson.get('geometry', geojson)
        if geometry['type'] == 'LineString':
            coords = geometry['coordinates']
            if coords[0] != coords[-1]:
                coords.append(coords[0])
        else:
            raise ValueError("Unsupported GeoJSON structure")
    
    return Polygon(coords)

def filter_points_within_polygon(xyz_data, polygon):
    """Filter XYZ points to only those within the polygon."""
    filtered = []
    for x, y, z in xyz_data:
        point = Point(x, y)
        if polygon.contains(point):
            filtered.append((x, y, z))
    return np.array(filtered)
    
    #return np.array(filtered_points)


def write_xyz_file(filename, points):
    """Write points to XYZ file"""
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def save_filtered_xyz(filename, data):
    """Save filtered XYZ data to a file."""
    with open(filename, 'w') as f:
        for x, y, z in data:
            f.write(f"{x} {y} {z}\n")
            
# execution part
def main():
    # Input files
    datos_xyz=read_xyz_file(r'D:\demar_3.nmp\demar_3\Export\Corredor_10Km_Interpolated_Full.xyz')#xyz_file = 'input.xyz'          # Replace with your XYZ file path
    polygon_vertices=read_polygon_file(r'D:\demar_3.nmp\demar_3\Export\polygon_mome_asccii.xyz')#polygon_file = 'polygon.txt'  # Replace with your polygon file path
    #polygon_vertices = load_geojson_polygon(r'D:\demar_3.nmp\demar_3\Export\mome_pl.geojson')#polygon_mome_asccii.xyz')
    output_file = r'D:\demar_3.nmp\demar_3\Export\filtered.xyz'
    # Check if we have enough vertices to form a polygon
    #if len(polygon_vertices) < 3:
    #    print("Error: Polygon needs at least 3 vertices")
    #    return
    # Filter points
    print("Filtering points...")
    filtered_data = filter_points_within_polygon(datos_xyz, polygon_vertices)
    #print(f"Filtered to {len(filtered_data)} points inside the polygon")
        
    # Save results
    print("Saving results...")
    save_filtered_xyz(output_file, filtered_data)
    print(f"Results saved to {output_file}")



# Extract X, Y, Z coordinates

if __name__ == "__main__":
    main()





print("--- %s seconds ---" % (time.time() - start_time))


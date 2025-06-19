import numpy as np
from shapely.geometry import Point, Polygon
import time
start_time = time.time()

def read_polygon(file_path):
    """Read polygon coordinates from a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse coordinates (assuming format is x,y,z)
    coords = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            coords.append((x, y))
    
    return coords

def read_xyz(file_path):
    """Read XYZ data from a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            data.append((x, y, z))
    
    return np.array(data)

def filter_points_in_polygon(xyz_data, polygon_coords):
    """Filter points that are inside the polygon"""
    polygon = Polygon(polygon_coords)
    filtered_points = []
    
    for point in xyz_data:
        shapely_point = Point(point[0], point[1])
        if polygon.contains(shapely_point):
            filtered_points.append(point)
    
    return np.array(filtered_points)

def save_filtered_data(filtered_data, output_file):
    """Save filtered data to a file"""
    with open(output_file, 'w') as f:
        for point in filtered_data:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    # Paths to your files
    polygon_file = r'D:\demar_3.nmp\demar_3\Export\poli_momefin.txt'
    xyz_file = r'D:\demar_3.nmp\demar_3\Export\Corredor_10Km_Interpolated_Full.xyz'  # Replace with your XYZ file path
    output_file = r'D:\demar_3.nmp\demar_3\Export\filtered_data.xyz'
    
    # Read data
    print("Reading polygon and XYZ data...")
    polygon_coords = read_polygon(polygon_file)
    xyz_data = read_xyz(xyz_file)
    print("Reading done")
    # Filter data
    print("filter the data ")
    filtered_data = filter_points_in_polygon(xyz_data, polygon_coords)
    print("Filtering done")
    
    # Save results
    print("Saving .........")
    save_filtered_data(filtered_data, output_file)
    
    print(f"Original points: {len(xyz_data)}")
    print(f"Filtered points: {len(filtered_data)}")
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    main()
    

print("--- %s seconds ---" % (time.time() - start_time))
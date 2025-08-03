import h5py
import os

h5_dir = 'C:/Users/srisr/Downloads/Knee-data-20250724T235546Z-1-002/Knee-data'  # Replace with your actual directory path
h5_file = os.listdir(h5_dir)[0]  # Pick the first .h5 file for inspection
file_path = os.path.join(h5_dir, h5_file)

with h5py.File(os.path.join(h5_dir, h5_file), 'r') as f:
    header = f['ismrmrd_header'][()].decode('utf-8')  # Decode if it's a byte string
    print(header)  # Look for coil-related metadata
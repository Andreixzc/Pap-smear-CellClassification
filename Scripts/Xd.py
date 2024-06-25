import os
import csv

def generate_csv_with_paths_and_labels(root_dir, output_csv):
    # List to hold paths and labels
    data = []

    # Walk through each directory and subdirectory
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            # Check if the file is an image (you can add more extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Get the full file path
                file_path = os.path.join(subdir, file)
                # Get the label which is the name of the parent directory
                label = os.path.basename(subdir)
                # Append to the list
                data.append([file_path, label])

    # Write the data to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Path", "Label"])  # Write the header
        writer.writerows(data)  # Write the data

    print(f"CSV file created: {output_csv}")

# Define the root directory containing the images and the output CSV file name
root_directory = '28-05-2024'
output_csv_file = 'output_images.csv'

# Generate the CSV
generate_csv_with_paths_and_labels(root_directory, output_csv_file)

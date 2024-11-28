import json
import os

# Directory containing the JSON files
save_directory = "C:/Users/Michalis/Desktop/ceid/HugginFace/Github/ExplainingBart/mm"

# List all JSON files in the directory
json_files = [f for f in os.listdir(save_directory) if f.endswith(".json")]

# Reformat each JSON file
for file_name in json_files:
    file_path = os.path.join(save_directory, file_name)

    # Read the original JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract the matrix (assumes original format is {"matrix": [...]})
    matrix = data["matrix"]

    # Save the reformatted JSON
    with open(file_path, "w") as f:
        json.dump(matrix, f)

    print(f"Reformatted: {file_path}")

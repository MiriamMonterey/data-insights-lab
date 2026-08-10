import kagglehub

# Download latest version
path = kagglehub.competition_download('titanic')

print("Path to competition files:", path)

# List files in the dataset
import os
for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
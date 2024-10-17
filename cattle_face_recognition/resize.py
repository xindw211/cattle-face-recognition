from PIL import Image
import os
# Adjust the image to 112x112

input_folder = "datasets/PNLS/train_cow"
output_folder = "datasets/train_cow"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        input_file_path = os.path.join(root, filename)
        relative_path = os.path.relpath(input_file_path, input_folder)
        output_file_path = os.path.join(output_folder, relative_path)
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with Image.open(input_file_path) as img:
            img = img.resize((112, 112))
            img.save(output_file_path)
print("处理完成")

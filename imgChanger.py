from PIL import Image
import os


def batch_resize(input_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = os.listdir(input_folder)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
    count = 0
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
        output_filename = f"1_{count:04d}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        img_resized.save(output_path)
        count += 1


if __name__ == "__main__":
    input_folder = r"C:\Users\HUAWEI\Desktop\AI‘\violence_224\aigcTest"
    output_folder = r"C:\Users\HUAWEI\Desktop\AI‘\violence_224\aigcTest2"
    target_width = 224
    target_height = 224

    batch_resize(input_folder, output_folder, target_width, target_height)

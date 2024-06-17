from PIL import Image
import os
import random


def add_noise_to_image(image_path, output_path, noise_size):
    image = Image.open(image_path)
    width, height = image.size
    noisy_image = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            pixel_color = image.getpixel((x, y))
            noise = (random.randint(-noise_size, noise_size),
                     random.randint(-noise_size, noise_size),
                     random.randint(-noise_size, noise_size))
            new_color = tuple(map(lambda i, j: max(0, min(255, i + j)), pixel_color, noise))
            noisy_image.putpixel((x, y), new_color)
    noisy_image.save(output_path)


def add_noise_to_images_in_folder(input_folder, output_folder, noise_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            add_noise_to_image(input_file_path, output_file_path, noise_size)


input_folder = r"C:\Users\HUAWEI\Desktop\AI‘\violence_224\test"
output_folder = r"C:\Users\HUAWEI\Desktop\AI‘\violence_224\noiseTest"
noise_size = 20
add_noise_to_images_in_folder(input_folder, output_folder, noise_size)

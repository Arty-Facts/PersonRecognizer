from pathlib import Path
from picamera import PiCamera
from time import sleep

import matplotlib.pyplot as plt

import numpy as np

camera = PiCamera()

camera.rotation = 180
camera.resolution = (256, 256)
camera.framerate = 15

working_directory = Path('.')
image_directory = working_directory/'pictures'
if not image_directory.is_dir():
    image_directory.mkdir()

# camera.start_preview()
# sleep(3)
output = np.empty((256,256,3), dtype=np.uint8)
# camera.capture(str(image_path))
for i in range(3):
    image_path = image_directory/f'bild{i}.png'
    camera.capture(output, 'rgb')
    # camera.stop_preview
    plt.imshow(output)
    plt.savefig(image_path)
    print(i)

def get_image():
    with PiCamera() as camera:
        camera.rotation = 180
        camera.resolution = (1920, 1080)
        camera.framerate = 15
        output = np.empty((1080,1920,3), dtype=np.uint8)
        camera.capture(output, 'rgb')
    return output
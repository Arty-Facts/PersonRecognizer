import torch 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from time import time, sleep
import cv2
import numpy as np
import skimage.io,  skimage.transform
import torchvision.transforms as transforms
from utils.utils import *
from PIL import Image

class Locate_ppl():
    def __init__(self, threshold=0.5, from_disk=False, path="images", save_img=False, back_bone="resnet50"):
        self.ssd_model = ssd()
        self.classes_to_labels = get_coco_object_dictionary()
        self.threshold = threshold
        self.ssd_model.eval()
        self.from_disk = from_disk
        self.set_path("images")
        self.set_path(path)
        self.save_img = save_img
        
    def set_path(self, path):
        _dir = Path(path)
        if not _dir.is_dir():
            _dir.mkdir()
        self.path = path

    def snap(self):
        if self.from_disk:
            image, org_size = next(get_image_disc(self.path))
        else:
            image, org_size = next(get_image())
            if self.save_img: #not recomended on a embeded device
                save_image(image, f"{self.path}/{get_time()}.jpg")
        inputs = [prepare_input(image)]
        tensor = prepare_tensor(inputs)
        with torch.no_grad():
            predicted_batch = self.ssd_model(tensor)
        # to do: vid mörker kraashar systemet
        results_per_input = decode_results(predicted_batch)
        fillterd_ouput = [pick_best(results, self.threshold) for results in results_per_input]
        ppl = []
        for bboxes, classes, confidences in fillterd_ouput:
            fig, ax = plt.subplots(1)
            # Show original, denormalized image...
            ax.imshow(image)
            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                x, y, w, h = [val * org_size for val in [left, bot, right - left, top - bot]]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y, "{} {:.0f}%".format(self.classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))

            for idx in range(len(bboxes)):
                if self.classes_to_labels[classes[idx] - 1] == "person":
                    left, bot, right, top = bboxes[idx]
                    fig, ax = plt.subplots(1)
                    location = [left, bot, abs(right - left), abs(top - bot)]
                    im = get_person(image,location, org_size)
                    # Show original, denormalized image...
                    im = rescale(im, 256, 256)
                    ppl.append(im)
                    ax.imshow(im)
        plt.show
        return ppl

    def process(self, image, org_size):
        inputs = [prepare_input(image)]
        tensor = prepare_tensor(inputs)
        with torch.no_grad():
            predicted_batch = self.ssd_model(tensor)
        results_per_input = decode_results(predicted_batch)
        fillterd_ouput = [pick_best(results, self.threshold) for results in results_per_input]
        ppl = []
        for bboxes, classes, confidences in fillterd_ouput:
            for idx in range(len(bboxes)):
                if self.classes_to_labels[classes[idx] - 1] == "person":
                    left, bot, right, top = bboxes[idx]
                    location = [left, bot, abs(right - left), abs(top - bot)]
                    im = get_person(image,location, org_size)
                    ppl.append(im)
        return ppl

    def get_images(self):
        if self.from_disk:
            for image, org_size in get_image_disc(self.path):
                yield self.process(image, org_size)
        else:
            for image, org_size in get_image():
                if self.save_img: #not recomended on a embeded device
                    save_image(image, f"{self.path}/{get_time()}.jpg")
                yield self.process(image, org_size)
    
    def __iter__(self):
        for ppl in self.get_images():
            yield ppl



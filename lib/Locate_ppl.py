import torch 
from matplotlib import pyplot as plt
import matplotlib.patches as patches
#from Gabby.camera import get_image
from time import time, sleep
import cv2
import numpy as np
import skimage.io,  skimage.transform
import torchvision.transforms as transforms
from utils.utils import *
from PIL import Image

class Locate_ppl():
    def __init__(self, threshold=0.5):
        precision = 'fp32'
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
        self.util = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.classes_to_labels = self.util.get_coco_object_dictionary()
        self.threshold = threshold
        self.ssd_model.eval()

    def snap(self):
        image, org_size = next(get_image())
        inputs = [prepare_input(image)]
        tensor = prepare_tensor(inputs)
        with torch.no_grad():
            predicted_batch = self.ssd_model(tensor)
        results_per_input = self.util.decode_results(predicted_batch)
        fillterd_ouput = [self.util.pick_best(results, self.threshold) for results in results_per_input]
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
    def get_images(self):
        for image, org_size in get_image():
            inputs = [prepare_input(image)]
            tensor = prepare_tensor(inputs)
            with torch.no_grad():
                predicted_batch = self.ssd_model(tensor)
            results_per_input = self.util.decode_results(predicted_batch)
            fillterd_ouput = [self.util.pick_best(results, self.threshold) for results in results_per_input]
            ppl = []
            for bboxes, classes, confidences in fillterd_ouput:
                for idx in range(len(bboxes)):
                    if self.classes_to_labels[classes[idx] - 1] == "person":
                        left, bot, right, top = bboxes[idx]
                        fig, ax = plt.subplots(1)
                        location = [left, bot, abs(right - left), abs(top - bot)]
                        im = get_person(image,location, org_size)
                        ppl.append(im)
            yield ppl
    
    def __iter__(self):
        for ppl in self.get_images():
            yield ppl



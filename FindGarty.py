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


precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
classes_to_labels = utils.get_coco_object_dictionary()
threshold = 0.5
ssd_model.eval()
while True:
    image, org_size = get_image()
    inputs = [prepare_input(image)]
    tensor = prepare_tensor(inputs)
    with torch.no_grad():
        predicted_batch = ssd_model(tensor)
    results_per_input = utils.decode_results(predicted_batch)
    fillterd_ouput = [utils.pick_best(results, threshold) for results in results_per_input]

    for bboxes, classes, confidences in fillterd_ouput:
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        ax.imshow(image)
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * org_size for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))

        for idx in range(len(bboxes)):
            if classes_to_labels[classes[idx] - 1] == "person":
                left, bot, right, top = bboxes[idx]
                fig, ax = plt.subplots(1)
                location = [left, bot, abs(right - left), abs(top - bot)]
                im = get_person(image,location, org_size)
                # Show original, denormalized image...
                ax.imshow(im)
    plt.show()
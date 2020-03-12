import torch 
import skimage.io,  skimage.transform
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from Gabby.camera import get_image
from time import time

precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
classes_to_labels = utils.get_coco_object_dictionary()

ssd_model.eval()
while True:
    start = time()
    inputs = [get_image(300,300)]
    print(f"{time()-start}s to get image")
    start = time()
    tensor = utils.prepare_tensor(inputs, precision == 'fp16')
    print(f"{time()-start}s to process image")
    with torch.no_grad():
        start = time()
        detections_batch = ssd_model(tensor)
        print(f"{time()-start}s to predict")
    start = time()
    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, 0.50) for results in results_per_input]

    for image_idx in range(len(best_results_per_input)):
        fig, ax = plt.subplots(1)
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))

        for idx in range(len(bboxes)):
            if classes_to_labels[classes[idx] - 1] == "person":
                left, bot, right, top = bboxes[idx]
                fig, ax = plt.subplots(1)
                x, y, w, h = [val for val in [left, bot, abs(right - left), abs(top - bot)]]
                # Show original, denormalized image...
                print(int(bot*300), int(top*300), int(left*300),int(right*300))
                im = image[max(0,int(bot*300)): max(0,int((bot+min(w, h))*300)), max(0,int(left*300)): max(0,int((left+min(w, h))*300))]
                ax.imshow(im)
    print(f"{time()-start}s gen image")
    plt.show()
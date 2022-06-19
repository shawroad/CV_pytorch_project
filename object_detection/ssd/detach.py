"""
@file   : detach.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-06-19$
"""
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
model = checkpoint['model']
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = model.to(device)
model.eval()

# Set detect transforms (It's important to be consistent with training)
resize = transforms.Resize((224, 224))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k):
    """
    Detect objects in an image with a trained tiny object detector, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: annotated image, a PIL Image
    """

    # Transform the image
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Post process, get the final detect objects from our tiny detector output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detect results to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_labels = det_labels[0].to('cpu').tolist()

    # Transform det_boxes to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels, for example: 12 -> dog, 15 -> person
    det_labels = [rev_label_map[l] for l in det_labels]

    # If no objects found, the detected labels will be set to ['0.']
    # you can find detail in tiny_detector.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate detect result on original image
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()
    for i in range(det_boxes.size(0)):
        box_location = det_boxes[i].tolist()

        # draw detect box
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])
        # a second rectangle at an offset of 1 pixel to increase line thickness

        # draw label Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1],
                            box_location[0] + text_size[0] + 4., box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
    label_map['background'] = 0
    rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    img_path = '../../../dataset/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
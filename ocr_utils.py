import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import cv2
import numpy as np
import pytesseract
import torchvision
from torchvision import transforms

# Load the pre-trained Faster R-CNN model (assume it's saved)
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 8 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(r"C:\Users\achus\Desktop\epoch_projects\note_ninja\arrow_resnet.pth"))
model.eval()

# Function to preprocess input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0), image

# Perform OCR using Tesseract
def extract_text(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

# Helper function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Combine overlapping and crowded boxes into a single encompassing box
def combine_overlapping_boxes(boxes, iou_threshold=0.01):
    combined_boxes = []

    while boxes:
        x_min, y_min, x_max, y_max = boxes.pop(0)
        merged = False

        for i, (cx_min, cy_min, cx_max, cy_max) in enumerate(combined_boxes):
            # Check if the boxes overlap
            if calculate_iou((x_min, y_min, x_max, y_max), (cx_min, cy_min, cx_max, cy_max)) > iou_threshold:
                # Combine the boxes into a single encompassing box
                combined_boxes[i] = (
                    min(x_min, cx_min),
                    min(y_min, cy_min),
                    max(x_max, cx_max),
                    max(y_max, cy_max)
                )
                merged = True
                break

        if not merged:
            combined_boxes.append((x_min, y_min, x_max, y_max))

    return combined_boxes

# Detect regions with enhanced box combination logic
def detect_regions(image_tensor, original_image):
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    threshold = 0.3

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()

    arrows, boxes_list, text_boxes = [], [], []

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            if label == 6:  # Assuming label 6 is arrows
                arrows.append((x_min, y_min, x_max, y_max))
            elif label in [1, 2, 4]:  # Assuming these labels represent generic boxes
                boxes_list.append((x_min, y_min, x_max, y_max))
            elif label == 5:  # Assuming label 5 is text boxes
                cropped_region = original_image.crop((x_min, y_min, x_max, y_max))
                text_boxes.append((cropped_region, (x_min, y_min, x_max, y_max)))

    # Filter and combine overlapping boxes
    filtered_arrows = combine_overlapping_boxes(arrows)
    filtered_boxes = combine_overlapping_boxes(boxes_list)

    # Combine overlapping text boxes
    text_boxes_coordinates = [box for _, box in text_boxes]
    combined_text_boxes = combine_overlapping_boxes(text_boxes_coordinates)

    combined_text_box_results = []
    for box in combined_text_boxes:
        x_min, y_min, x_max, y_max = box
        cropped_region = original_image.crop((x_min, y_min, x_max, y_max))
        combined_text_box_results.append((cropped_region, box))

    return filtered_arrows, filtered_boxes, combined_text_box_results

# Function to calculate the Euclidean distance between two box centers
def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5

# Filter text boxes based on proximity to arrows or diagram boxes
def filter_text_boxes(text_boxes, arrows, boxes_list, proximity_threshold):
    retained_text_boxes = []
    removed_text_boxes = []

    for text_box, coords in text_boxes:
        is_close = False
        for box in arrows + boxes_list:
            distance = calculate_distance(coords, box)
            if distance <= proximity_threshold:
                is_close = True
                break
        if is_close:
            retained_text_boxes.append((text_box, coords))
        else:
            removed_text_boxes.append((text_box, coords))

    return retained_text_boxes, removed_text_boxes

# Main processing function
def process_handwritten_script(image, proximity_threshold=70):
    image_tensor, original_image = preprocess_image(image)

    arrows, boxes_list, text_boxes = detect_regions(image_tensor, original_image)

    # Filter text boxes based on proximity threshold
    retained_text_boxes, removed_text_boxes = filter_text_boxes(
        [(crop, box) for crop, box in text_boxes], arrows, boxes_list, proximity_threshold
    )

    extracted_text_results = []
    if retained_text_boxes:
        for idx, (text_image, (x_min, y_min, x_max, y_max)) in enumerate(retained_text_boxes):
            ocr_result = extract_text(text_image)
            extracted_text_results.append({
                "text": ocr_result.strip(),
                "bbox": (x_min, y_min, x_max, y_max)
            })

    removed_text_results = []
    if removed_text_boxes:
        for idx, (text_image, (x_min, y_min, x_max, y_max)) in enumerate(removed_text_boxes):
            ocr_result = extract_text(text_image)
            removed_text_results.append({
                "text": ocr_result.strip(),
                "bbox": (x_min, y_min, x_max, y_max)
            })

    # Create mask
    mask = np.ones(original_image.size[::-1], dtype=np.uint8) * 255  # White mask
    for x_min, y_min, x_max, y_max in arrows + boxes_list:
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=0, thickness=-1)
    for _, (x_min, y_min, x_max, y_max) in retained_text_boxes:
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), color=0, thickness=-1)

    # Apply mask to original image
    masked_image = Image.fromarray(cv2.bitwise_and(np.array(original_image), np.array(original_image), mask=mask))
    non_diagram_text = extract_text(masked_image)


    output = {
        "arrows": arrows,
        "boxes": boxes_list,
        "retained_text_boxes": extracted_text_results,
        "text_outside_diagrams": non_diagram_text.strip()
    }

    return output

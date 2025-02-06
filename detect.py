import numpy as np
import subprocess
import os
import time
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    FCOS_ResNet50_FPN_Weights
)

# tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# for YOLO model
from ultralytics import YOLO
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Import segmentation Models
from segmentation_models import run_segmentation

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "TV",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to calculate image size in bits
def get_image_size_in_bits(image_path):
    file_size_bytes = os.path.getsize(image_path)  # File size in bytes
    file_size_bits = file_size_bytes * 8  # Convert to bits (1 byte = 8 bits)
    return file_size_bits

# Function to get real-time power consumption (for Nvidia GPUs)
def get_power_usage_nvidia():
    try:
        # Execute nvidia-smi to get power consumption
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
        power_watts = float(output.decode('utf-8').strip())
        return power_watts
    except Exception as e:
        print(f"Error fetching power consumption: {e}")
        return 0  # Return 0 if unable to fetch
    
# Function to calculate energy required
def calculate_energy(power_watts, latency_seconds):
    return power_watts * latency_seconds  # Energy in joules (watts * seconds)


# TorchVision models
def load_model(model_name="fasterrcnn"):
    if model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
    elif model_name == "fasterrcnn_v2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "maskrcnn_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "keypointrcnn":
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "ssd":
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    elif model_name == "ssdlite":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    elif model_name == "resnet18_detection":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "fcos_resnet50":
        model = torchvision.models.detection.fcos_resnet50_fpn(weights="DEFAULT")
    elif model_name == "fasterrcnn_inceptionv2":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif model_name == "maskrcnn_inceptionv2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    elif model_name == "ssd_mobilenet_v1":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    elif model_name == "ssd_resnet34":
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.eval()
    return model


# Function to load TensorFlow models
def load_tf_model(model_name):
    if model_name == "ssd_mobilenet_v2":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    elif model_name == "ssd_mobilenet_v1":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1")
    elif model_name == "ssd_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/ssd_resnet50_v1_fpn_640x640/1")
        
    # Faster R-CNN models
    elif model_name == "faster_rcnn_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1")
    elif model_name == "faster_rcnn_inception":
        model = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")
        
    # EfficientDet models
    elif model_name == "efficientdet_d0":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")
    elif model_name == "efficientdet_d1":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d1/1")
    elif model_name == "efficientdet_d2":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d2/1")
    elif model_name == "efficientdet_d3":
        model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d3/1")
        
    # RetinaNet models
    elif model_name == "retinanet":
        model = hub.load("https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1")
  
    # CenterNet models
    elif model_name == "centernet_hourglass":
        model = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
    elif model_name == "centernet_resnet50":
        model = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1")
    
    else:
        raise ValueError("Model name not recognized. Please choose a valid model.")
    
    return model


def load_yo_model(model_name):
    if model_name == "yolov5n":
        model = YOLO("yolov5s.pt")
    elif model_name == "yolov5s":
        model = YOLO('yolov5s.pt')
    elif model_name == "yolov5m":
        model = YOLO('yolov5m.pt')
    elif model_name == "yolov5l":
        model = YOLO('yolov5l.pt')
    elif model_name == "yolov5x":
        model = YOLO("yolov5x.pt")
    # yolo 8
    elif model_name == "yolov8n":
        model = YOLO("yolov8n.pt")
    elif model_name == "yolov8s":
        model = YOLO("yolov8s.pt")
    elif model_name == "yolov8m":
        model = YOLO("yolov8m.pt")
    elif model_name == "yolov8l":
        model = YOLO("yolov8l.pt")
    elif model_name == "yolov8x":
        model = YOLO("yolov8x.pt")

    # yolo 9
    elif model_name == "yolov9t":
        model = YOLO("yolov9t.pt")
    elif model_name == "yolov9s":
        model = YOLO("yolov9s.pt")
    elif model_name == "yolov9m":
        model = YOLO("yolov9m.pt")
    elif model_name == "yolov9c":
        model = YOLO("yolov9c.pt")
    elif model_name == "yolov9e":
        model = YOLO("yolov9e.pt")

    
    # yolo 10
    elif model_name == "yolov10n":
        model = YOLO("yolov10n.pt")
    elif model_name == "yolov10s":
        model = YOLO("yolov10s.pt")
    elif model_name == "yolov10m":
        model = YOLO("yolov10m.pt")
    elif model_name == "yolov10l":
        model = YOLO("yolov10l.pt")
    elif model_name == "yolov10x":
        model = YOLO("yolov10x.pt")
        # Yolo 11
    elif model_name == "yolo11n":
        model = YOLO("yolo11n.pt")
    elif model_name == "yolo11s":
        model = YOLO("yolo11s.pt")
    elif model_name == "yolo11m":
        model = YOLO("yolo11m.pt")
    elif model_name == "yolo11l":
        model = YOLO("yolo11l.pt")
    elif model_name == "yolo11x":
        model = YOLO("yolo11x.pt")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


# Function to calculate latency and response time
def calculate_latency(start_time, end_time):
    return end_time - start_time


def run_yo_model(image_path, model_name):
    model = load_yo_model(model_name)
    start_time = time.time()

    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    # Perform inference on the image
    results = model(image)

    end_time = time.time()
    # 
    latency = end_time - start_time

    detected_objects = []
    total_confidence = 0
    num_detections = 0

    for result in results:
        for box in result.boxes:
            obj_class = result.names[int(box.cls)]
            confidence = box.conf.item()
            bbox = box.xyxy.tolist()[0]  # x_min, y_min, x_max, y_max

            # Transform bbox into the required format
            x_min, y_min, x_max, y_max = bbox
            detected_objects.append({
                'label': obj_class,
                'confidence': confidence,
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
            total_confidence += confidence
            num_detections += 1

    # Calculate accuracy
    overall_accuracy = total_confidence / num_detections if num_detections > 0 else 0

    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)
    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return detected_objects, latency, overall_accuracy, throughput, energy_required, power_watts


# Function to run the model and perform object detection
def run_tf_model(image_path, model_name):
    # Load the specified TensorFlow model
    model = load_tf_model(model_name)
    # Load and preprocess the image
    if image_path.startswith('http'):  # If the input is a URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    # Convert image to tensor and normalize
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    start_time = time.time()
    # Perform object detection
    detections = model(image_tensor)
    # Stop the inference time
    end_time = time.time()
    latency = calculate_latency(start_time, end_time)
    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation
    image_width, image_height = image.size
    for i in range(len(detections['detection_scores'][0])):
        score = detections['detection_scores'][0][i].numpy()
        if score > 0.5:  # Confidence threshold
            box = detections['detection_boxes'][0][i].numpy().tolist()  # Convert tensor to list
            label_id = int(detections['detection_classes'][0][i].numpy())  # Convert to Python int
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
            x_min = int(box[1] * image_width)
            y_min = int(box[0] * image_height)
            x_max = int(box[3] * image_width)
            y_max = int(box[2] * image_height)
            detected_objects.append({
                'label': label_name,
                'confidence': score,
                'x': x_min,
                'y': y_min,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
            accuracy += score  # Sum the confidence for accuracy calculation
    # Calculate average accuracy
    accuracy /= len(detected_objects) if detected_objects else 1  # Avoid division by zero

    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)
    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return detected_objects, latency, accuracy, throughput, energy_required, power_watts


# Function to run the model and perform object detection
def run_model(image_path, model_name="fasterrcnn"):
    if model_name[:2] == 'sg':
        model_ = model_name[3::]
        return run_segmentation(image_path=image_path,model_name=model_)
    elif model_name[:2] == 'tf':
        model_ = model_name[3::]
        return run_tf_model(image_path=image_path,model_name=model_)
    elif model_name[:2] == 'yo':
        model_ = model_name[3::]
        return run_yo_model(image_path=image_path,model_name=model_)

    # Load the specified model
    model = load_model(model_name)

    # Load and preprocess the image
    if image_path.startswith('http'):  # If the input is a URL
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    print(image_tensor.shape)
    start_time = time.time()
    
    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Stop the inference time
    end_time = time.time()
    latency = calculate_latency(start_time, end_time)

    # Process the predictions
    detected_objects = []
    accuracy = 0.0  # Placeholder for accuracy calculation

    if 'scores' in predictions[0] and 'labels' in predictions[0] and 'boxes' in predictions[0]:
        for i, score in enumerate(predictions[0]['scores'].tolist()):  # Convert to list for compatibility
            if score > 0.5:  # Confidence threshold
                box = predictions[0]['boxes'][i].tolist()  # Convert tensor to list
                label_id = predictions[0]['labels'][i].item()  # Convert to Python int
                
                if type(label_id) == int:
                    # Convert label_id to string using the COCO class names
                    label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]  # Convert to string class name
                    # print(f"Detected object label: {label_name} (confidence: {score})")
                else:
                    label_name = label_id
                    
                detected_objects.append({
                    'label': label_name,  # Store the string label name
                    'confidence': score,
                    'x': box[0],  # x_min
                    'y': box[1],  # y_min
                    'width': box[2] - box[0],  # width
                    'height': box[3] - box[1]  # height
                })
                accuracy += score  # Sum the confidence for accuracy calculation

    # Calculate average accuracy
    accuracy /= len(detected_objects) if detected_objects else 1  # Avoid division by zero
    # Get dynamic power consumption from Nvidia GPU
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Get image size in bits
    image_size_bits = get_image_size_in_bits(image_path)

    throughput_result = float(f"{image_size_bits / latency if latency > 0 else 0:.2f}")
    throughput = throughput_result / 1000

    return detected_objects, latency, accuracy, throughput, energy_required, power_watts


# if __name__ == "__main__":
#     image_path = "ppc-0603.jpeg"  # Replace with your image path or URL
#     model_name = 'ssd_resnet34'
#     results, latency, overall_accuracy, throughput, energy_required, power_watts = run_model(image_path, model_name)
#     # print(f"Detected objects: {results}")
#     print(f"Latency: {latency:.2f} seconds")
#     print(f"Accuracy: {overall_accuracy:.2f}")
#     print(f"throughput: {throughput:.2f}")
#     print(f"energy_required: {energy_required:.2f}")
#     print(f"power_watts: {power_watts:.2f}")

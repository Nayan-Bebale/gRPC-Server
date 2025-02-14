import cv2
import numpy as np
import time
import os
import torch
import subprocess
import torchvision
from torchvision import transforms
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights, 
                                          MaskRCNN_ResNet50_FPN_V2_Weights,
                                            KeypointRCNN_ResNet50_FPN_Weights)
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import segmentation_models_pytorch as smp
from PIL import Image
from io import BytesIO
import requests
from ultralytics import YOLO, SAM, FastSAM, RTDETR
from torchvision.transforms.functional import to_tensor


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

# # Define a mapping of unique names to model-encoder combinations
MODEL_ENCODER_MAP = {
    "Unet_vgg11": ("Unet", "vgg11"),
    "Unet_vgg13": ("Unet", "vgg13"),
    "Unet_vgg16": ("Unet", "vgg16"),
    "Unet_vgg19": ("Unet", "vgg19"),
    "Unet_vgg11_bn": ("Unet", "vgg11_bn"),
    "Unet_vgg13_bn": ("Unet", "vgg13_bn"),
    "Unet_vgg16_bn": ("Unet", "vgg16_bn"),
    "Unet_vgg19_bn": ("Unet", "vgg19_bn"),
    "Unet_mit_b0": ("Unet", "mit_b0"),
    "Unet_mit_b1": ("Unet", "mit_b1"),
    "Unet_mit_b2": ("Unet", "mit_b2"),
    "Unet_mit_b3": ("Unet", "mit_b3"),
    "Unet_mit_b4": ("Unet", "mit_b4"),
    "Unet_mit_b5": ("Unet", "mit_b5"),
    "Unet_mobileone_s0": ("Unet", "mobileone_s0"),
    "Unet_mobileone_s1": ("Unet", "mobileone_s1"),
    "Unet_mobileone_s2": ("Unet", "mobileone_s2"),
    "Unet_mobileone_s3": ("Unet", "mobileone_s3"),
    "Unet_mobileone_s4": ("Unet", "mobileone_s4"),
    "Unet_resnet18": ("Unet", "resnet18"),
    "Unet_resnet34": ("Unet", "resnet34"),
    "Unet_resnet50": ("Unet", "resnet50"),
    "Unet_resnet101": ("Unet", "resnet101"),
    "Unet_resnet152": ("Unet", "resnet152"),
    "Unet_mobilenetv2": ("Unet", "mobilenet_v2"),
    "Unet_efficientnetb0": ("Unet", "efficientnet-b0"),
    "Unet_efficientnetb1": ("Unet", "efficientnet-b1"),
    "Unet_efficientnetb2": ("Unet", "efficientnet-b2"),
    "Unet_efficientnetb3": ("Unet", "efficientnet-b3"),
    "Unet_efficientnetb4": ("Unet", "efficientnet-b4"),
    "Unet_efficientnetb5": ("Unet", "efficientnet-b5"),
    "Unet_efficientnetb6": ("Unet", "efficientnet-b6"),
    "Unet_efficientnetb6": ("Unet", "efficientnet-b7"),
    "Unet_timm-efficientnetb0": ("Unet", "timm-efficientnet-b0"),
    "Unet_timm-efficientnetb1": ("Unet", "timm-efficientnet-b1"),
    "Unet_timm-efficientnetb2": ("Unet", "timm-efficientnet-b2"),
    "Unet_timm-efficientnetb3": ("Unet", "timm-efficientnet-b3"),
    "Unet_timm-efficientnetb4": ("Unet", "timm-efficientnet-b4"),
    "Unet_timm-efficientnetb5": ("Unet", "timm-efficientnet-b5"),
    "Unet_timm-efficientnetb6": ("Unet", "timm-efficientnet-b6"),
    "Unet_timm-efficientnetb7": ("Unet", "timm-efficientnet-b7"),
    "Unet_timm-efficientnetb8": ("Unet", "timm-efficientnet-b8"),
    "Unet_timm-tf_efficientnet_lite0": ("Unet", "timm-tf_efficientnet_lite0"),
    "Unet_timm-tf_efficientnet_lite1": ("Unet", "timm-tf_efficientnet_lite1"),
    "Unet_timm-tf_efficientnet_lite2": ("Unet", "timm-tf_efficientnet_lite2"),
    "Unet_timm-tf_efficientnet_lite3": ("Unet", "timm-tf_efficientnet_lite3"),
    "Unet_timm-tf_efficientnet_lite4": ("Unet", "timm-tf_efficientnet_lite4"),
    

    "DeepLabV3_resnet18": ("DeepLabV3", "resnet18"),
    "DeepLabV3_resnet34": ("DeepLabV3", "resnet34"),
    "DeepLabV3_resnet50": ("DeepLabV3", "resnet50"),
    "DeepLabV3_resnet101": ("DeepLabV3", "resnet101"),
    "DeepLabV3_resnet152": ("DeepLabV3", "resnet152"),
    "DeepLabV3_mobilenetv2": ("DeepLabV3", "mobilenet_v2"),
    "DeepLabV3_mobileone_s0": ("DeepLabV3", "mobileone_s0"),
    "DeepLabV3_mobileone_s1": ("DeepLabV3", "mobileone_s1"),
    "DeepLabV3_mobileone_s2": ("DeepLabV3", "mobileone_s2"),
    "DeepLabV3_mobileone_s3": ("DeepLabV3", "mobileone_s3"),
    "DeepLabV3_mobileone_s4": ("DeepLabV3", "mobileone_s4"),
    "DeepLabV3_efficientnetb0": ("DeepLabV3", "efficientnet-b0"),
    "DeepLabV3_efficientnetb1": ("DeepLabV3", "efficientnet-b1"),
    "DeepLabV3_efficientnetb2": ("DeepLabV3", "efficientnet-b2"),
    "DeepLabV3_efficientnetb3": ("DeepLabV3", "efficientnet-b3"),
    "DeepLabV3_efficientnetb4": ("DeepLabV3", "efficientnet-b4"),
    "DeepLabV3_efficientnetb5": ("DeepLabV3", "efficientnet-b5"),
    "DeepLabV3_efficientnetb6": ("DeepLabV3", "efficientnet-b6"),
    "DeepLabV3_efficientnetb7": ("DeepLabV3", "efficientnet-b7"),
    "DeepLabV3_timm-efficientnetb0": ("DeepLabV3", "timm-efficientnet-b0"),
    "DeepLabV3_timm-efficientnetb1": ("DeepLabV3", "timm-efficientnet-b1"),
    "DeepLabV3_timm-efficientnetb2": ("DeepLabV3", "timm-efficientnet-b2"),
    "DeepLabV3_timm-efficientnetb3": ("DeepLabV3", "timm-efficientnet-b3"),
    "DeepLabV3_timm-efficientnetb4": ("DeepLabV3", "timm-efficientnet-b4"),
    "DeepLabV3_timm-efficientnetb5": ("DeepLabV3", "timm-efficientnet-b5"),
    "DeepLabV3_timm-efficientnetb6": ("DeepLabV3", "timm-efficientnet-b6"),
    "DeepLabV3_timm-efficientnetb7": ("DeepLabV3", "timm-efficientnet-b7"),
    "DeepLabV3_timm-efficientnetb8": ("DeepLabV3", "timm-efficientnet-b8"),
    "DeepLabV3_timm-tf_efficientnet_lite0": ("DeepLabV3", "timm-tf_efficientnet_lite0"),
    "DeepLabV3_timm-tf_efficientnet_lite1": ("DeepLabV3", "timm-tf_efficientnet_lite1"),
    "DeepLabV3_timm-tf_efficientnet_lite2": ("DeepLabV3", "timm-tf_efficientnet_lite2"),
    "DeepLabV3_timm-tf_efficientnet_lite3": ("DeepLabV3", "timm-tf_efficientnet_lite3"),


    "FPN_vgg11": ("FPN", "vgg11"),
    "FPN_vgg13": ("FPN", "vgg13"),
    "FPN_vgg16": ("FPN", "vgg16"),
    "FPN_vgg19": ("FPN", "vgg19"),
    "FPN_vgg11_bn": ("FPN", "vgg11_bn"),
    "FPN_vgg13_bn": ("FPN", "vgg13_bn"),
    "FPN_vgg16_bn": ("FPN", "vgg16_bn"),
    "FPN_vgg19_bn": ("FPN", "vgg19_bn"),
    "FPN_mit_b0": ("FPN", "mit_b0"),
    "FPN_mit_b1": ("FPN", "mit_b1"),
    "FPN_mit_b2": ("FPN", "mit_b2"),
    "FPN_mit_b3": ("FPN", "mit_b3"),
    "FPN_mit_b4": ("FPN", "mit_b4"),
    "FPN_mit_b5": ("FPN", "mit_b5"),
    "FPN_resnet18": ("FPN", "resnet18"),
    "FPN_resnet34": ("FPN", "resnet34"),
    "FPN_resnet50": ("FPN", "resnet50"),
    "FPN_resnet101": ("FPN", "resnet101"),
    "FPN_resnet152": ("FPN", "resnet152"),
    "FPN_mobilenetv2": ("FPN", "mobilenet_v2"),
    "FPN_mobileone_s0": ("FPN", "mobileone_s0"),
    "FPN_mobileone_s1": ("FPN", "mobileone_s1"),
    "FPN_mobileone_s2": ("FPN", "mobileone_s2"),
    "FPN_mobileone_s3": ("FPN", "mobileone_s3"),
    "FPN_mobileone_s4": ("FPN", "mobileone_s4"),
    "FPN_efficientnetb0": ("FPN", "efficientnet-b0"),
    "FPN_efficientnetb1": ("FPN", "efficientnet-b1"),
    "FPN_efficientnetb2": ("FPN", "efficientnet-b2"),
    "FPN_efficientnetb3": ("FPN", "efficientnet-b3"),
    "FPN_efficientnetb4": ("FPN", "efficientnet-b4"),
    "FPN_efficientnetb5": ("FPN", "efficientnet-b5"),
    "FPN_efficientnetb6": ("FPN", "efficientnet-b6"),
    "FPN_efficientnetb7": ("FPN", "efficientnet-b7"),
    "FPN_timm-efficientnetb0": ("FPN", "timm-efficientnet-b0"),
    "FPN_timm-efficientnetb1": ("FPN", "timm-efficientnet-b1"),
    "FPN_timm-efficientnetb2": ("FPN", "timm-efficientnet-b2"),
    "FPN_timm-efficientnetb3": ("FPN", "timm-efficientnet-b3"),
    "FPN_timm-efficientnetb4": ("FPN", "timm-efficientnet-b4"),
    "FPN_timm-efficientnetb5": ("FPN", "timm-efficientnet-b5"),
    "FPN_timm-efficientnetb6": ("FPN", "timm-efficientnet-b6"),
    "FPN_timm-efficientnetb7": ("FPN", "timm-efficientnet-b7"),
    "FPN_timm-efficientnetb8": ("FPN", "timm-efficientnet-b8"),
    "FPN_timm-tf_efficientnet_lite0": ("FPN", "timm-tf_efficientnet_lite0"),
    "FPN_timm-tf_efficientnet_lite1": ("FPN", "timm-tf_efficientnet_lite1"),
    "FPN_timm-tf_efficientnet_lite2": ("FPN", "timm-tf_efficientnet_lite2"),
    "FPN_timm-tf_efficientnet_lite3": ("FPN", "timm-tf_efficientnet_lite3"),
    "FPN_timm-tf_efficientnet_lite4": ("FPN", "timm-tf_efficientnet_lite4"),


    "PSPNet_vgg11": ("PSPNet", "vgg11"),
    "PSPNet_vgg13": ("PSPNet", "vgg13"),
    "PSPNet_vgg16": ("PSPNet", "vgg16"),
    "PSPNet_vgg19": ("PSPNet", "vgg19"),
    "PSPNet_vgg11_bn": ("PSPNet", "vgg11_bn"),
    "PSPNet_vgg13_bn": ("PSPNet", "vgg13_bn"),
    "PSPNet_vgg16_bn": ("PSPNet", "vgg16_bn"),
    "PSPNet_vgg19_bn": ("PSPNet", "vgg19_bn"),
    "PSPNet_mit_b0": ("PSPNet", "mit_b0"),
    "PSPNet_mit_b1": ("PSPNet", "mit_b1"),
    "PSPNet_mit_b2": ("PSPNet", "mit_b2"),
    "PSPNet_mit_b3": ("PSPNet", "mit_b3"),
    "PSPNet_mit_b4": ("PSPNet", "mit_b4"),
    "PSPNet_mit_b5": ("PSPNet", "mit_b5"),
    "PSPNet_resnet18": ("PSPNet", "resnet18"),
    "PSPNet_resnet34": ("PSPNet", "resnet34"),
    "PSPNet_resnet50": ("PSPNet", "resnet50"),
    "PSPNet_resnet101": ("PSPNet", "resnet101"),
    "PSPNet_resnet152": ("PSPNet", "resnet152"),
    "PSPNet_mobilenetv2": ("PSPNet", "mobilenet_v2"),
    "PSPNet_mobileone_s0": ("PSPNet", "mobileone_s0"),
    "PSPNet_mobileone_s1": ("PSPNet", "mobileone_s1"),
    "PSPNet_mobileone_s2": ("PSPNet", "mobileone_s2"),
    "PSPNet_mobileone_s3": ("PSPNet", "mobileone_s3"),
    "PSPNet_mobileone_s4": ("PSPNet", "mobileone_s4"),
    "PSPNet_efficientnetb0": ("PSPNet", "efficientnet-b0"),
    "PSPNet_efficientnetb1": ("PSPNet", "efficientnet-b1"),
    "PSPNet_efficientnetb2": ("PSPNet", "efficientnet-b2"),
    "PSPNet_efficientnetb3": ("PSPNet", "efficientnet-b3"),
    "PSPNet_efficientnetb4": ("PSPNet", "efficientnet-b4"),
    "PSPNet_efficientnetb5": ("PSPNet", "efficientnet-b5"),
    "PSPNet_efficientnetb6": ("PSPNet", "efficientnet-b6"),
    "PSPNet_efficientnetb7": ("PSPNet", "efficientnet-b7"),
    "PSPNet_timm-efficientnetb0": ("PSPNet", "timm-efficientnet-b0"),
    "PSPNet_timm-efficientnetb1": ("PSPNet", "timm-efficientnet-b1"),
    "PSPNet_timm-efficientnetb2": ("PSPNet", "timm-efficientnet-b2"),
    "PSPNet_timm-efficientnetb3": ("PSPNet", "timm-efficientnet-b3"),
    "PSPNet_timm-efficientnetb4": ("PSPNet", "timm-efficientnet-b4"),
    "PSPNet_timm-efficientnetb5": ("PSPNet", "timm-efficientnet-b5"),
    "PSPNet_timm-efficientnetb6": ("PSPNet", "timm-efficientnet-b6"),
    "PSPNet_timm-efficientnetb7": ("PSPNet", "timm-efficientnet-b7"),
    "PSPNet_timm-efficientnetb8": ("PSPNet", "timm-efficientnet-b8"),
    "PSPNet_timm-tf_efficientnet_lite0": ("PSPNet", "timm-tf_efficientnet_lite0"),
    "PSPNet_timm-tf_efficientnet_lite1": ("PSPNet", "timm-tf_efficientnet_lite1"),
    "PSPNet_timm-tf_efficientnet_lite2": ("PSPNet", "timm-tf_efficientnet_lite2"),
    "PSPNet_timm-tf_efficientnet_lite3": ("PSPNet", "timm-tf_efficientnet_lite3"),
    "PSPNet_timm-tf_efficientnet_lite4": ("PSPNet", "timm-tf_efficientnet_lite4"),
    "PSPNet_timm-res2next50": ("PSPNet", "timm-res2next50"), 
    "PSPNet_timm-regnetx_320": ("PSPNet", 'timm-regnetx_320'), 
}


# MODEL_ENCODER_MAP = {
#     "PSPNet_timm-regnetx_320": ("PSPNet", 'timm-regnetx_320'),
    
# }

#################################################################################### Get parameters

# Function to calculate energy required
def calculate_energy(power_watts, latency_seconds):
    return power_watts * latency_seconds  # Energy in joules (watts * seconds)

# Function to calculate latency
def calculate_latency(start_time, end_time):
    return end_time - start_time

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


################################################################################# Load Models

# Function to load PyTorch segmentation models
def load_segmentation_model(model_name="maskrcnn"):
    if model_name == "maskrcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    elif model_name == "maskrcnn_v2":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    elif model_name == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    elif model_name == "deeplabv3_resnet101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    elif model_name == "fcn_resnet50":
        model = torchvision.models.segmentation.fcn_resnet50(weights=torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT)
    elif model_name == "fcn_resnet101":
        model = torchvision.models.segmentation.fcn_resnet101(weights=torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT)
    elif model_name == "lraspp_mobilenet_v3_large":
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
    elif model_name == "deeplabv3_mobilenet_v3_large":
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
    elif model_name == "retinanet_resnet50_fpn":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported segmentation model: {model_name}")
    model.eval()
    return model

# Function to load YOLO models with segmentation
def load_yolo_segmentation_model(model_name="yolov8n_seg"):
    if model_name == "yolov8n_seg":
        model = YOLO("yolov8n-seg.pt")  # Nano
    elif model_name == "yolov8s_seg":
        model = YOLO("yolov8s-seg.pt")  # Small
    elif model_name == "yolov8m_seg":
        model = YOLO("yolov8m-seg.pt")  # Medium
    elif model_name == "yolov8l_seg":
        model = YOLO("yolov8l-seg.pt")  # Large
    elif model_name == "yolov8x_seg":
        model = YOLO("yolov8x-seg.pt")  # Extra large


    elif model_name == "yolov11n_seg":
        model = YOLO("yolo11n-seg.pt")  # Nano
    elif model_name == "yolov11s_seg":
        model = YOLO("yolo11s-seg.pt")  # Small
    elif model_name == "yolov11m_seg":
        model = YOLO("yolo11m-seg.pt")  # Medium
    elif model_name == "yolov11l_seg":
        model = YOLO("yolo11l-seg.pt")  # Large
    elif model_name == "yolov11x_seg":
        model = YOLO("yolo11x-seg.pt")  # Extra large


    elif model_name == ("sam_l"):
        model = SAM("sam_l.pt")
    elif model_name == ("sam_b"):
        model = SAM("sam_b.pt") 
    elif model_name == ("mobile_sam"):
        model = SAM("mobile_sam.pt") 
    elif model_name == ("sam2_t"):
        model = SAM("sam2_t.pt")
    elif model_name == ("sam2_s"):
        model = SAM("sam2_s.pt")
    elif model_name == ("sam2_b"):
        model = SAM("sam2_b.pt")
    elif model_name == ("sam2_l"):
        model = SAM("sam2_l.pt")

    elif model_name == ("FastSAM-s"):
        model = FastSAM("FastSAM-s.pt")
    elif model_name == ("FastSAM-x"):
        model = FastSAM("FastSAM-x.pt")

    else:
        raise ValueError(f"Unsupported YOLO segmentation model: {model_name}")
    return model

def load_segmentation_model_sm(model_name):
    if model_name not in MODEL_ENCODER_MAP:
        raise ValueError(f"Unsupported model name: {model_name}")

    base_model, encoder_name = MODEL_ENCODER_MAP[model_name]
    print(f"Loading model: {base_model} with encoder: {encoder_name}")

    model = getattr(smp, base_model)(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        classes=1,
        activation="sigmoid"
    )
    return model

################################################################################## Process images

#  Function to preprocess image
def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(input_image).unsqueeze(0)

# Preprocess the input image
def preprocess_image_sm(image_path, input_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

################################################################################## Process Segmentation for predictions

def process_segmentation_predictions_ch(predictions, image_shape):
    detected_objects = []
    masks = predictions[0].get('masks', None)
    scores = predictions[0].get('scores', None)
    labels = predictions[0].get('labels', None)
    boxes = predictions[0].get('boxes', None)
    height, width = image_shape

    if scores is not None and masks is not None and labels is not None:
        for i, score in enumerate(scores.tolist()):
            if score > 0.5:  # Confidence threshold
                try:
                    mask = masks[i][0].mul(255).byte().cpu().numpy()  # Convert to binary mask
                    label_id = labels[i].item()
                    label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
                    box = boxes[i].tolist()
                    x_min, y_min, x_max, y_max = box
                    detected_objects.append({
                        'label': label_name,
                        'confidence': score,
                        'x': x_min,
                        'y': y_min,
                        'width': x_max - x_min,
                        'height': y_max - y_min,
                        'mask': mask  # Segmentation mask
                    })
                except IndexError:
                    print(f"IndexError: Issue with label ID {label_id} or prediction format")
    else:
        print("Missing keys in predictions:", predictions[0].keys())
    return detected_objects, 0.0  # Return default accuracy if no objects detected


def process_segmentation_predictions(predictions, image_shape):
    detected_objects = []
    height, width = image_shape
    try:
        # DeepLabV3 outputs predictions in the 'out' key
        segmentation_output = predictions['out']  # Get the main output
        segmentation_map = torch.argmax(segmentation_output.squeeze(), dim=0).detach().cpu().numpy()

        # For calculating accuracy (example approach)
        unique_labels = torch.unique(segmentation_output.argmax(dim=1))
        accuracy = len(unique_labels) / segmentation_output.shape[1]  # Dummy metric for demo

        print("Come close")
        detected_objects.append({
            'segmentation_map': segmentation_map,  # Segmentation map
            'height': height,
            'width': width
        })

        return detected_objects, accuracy
    except TypeError:
        return process_segmentation_predictions_ch(predictions, image_shape)
    

# Process predictions and handle ground truth absence
def process_segmentation_predictions_sm(predictions, image_shape, ground_truth_path=None):
    pred_mask = (predictions.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Accuracy as pixel ratio if ground truth is absent
    accuracy = (pred_mask.sum() / (image_shape[0] * image_shape[1])) 

    # IoU if ground truth is provided
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            accuracy = (intersection / union) if union > 0 else 0

    return pred_mask, accuracy


############################################################################ Run models

# run yolo models
def run_yolo_models(image_path, model_name):
    model = load_yolo_segmentation_model(model_name)
    start_time = time.time()

    # Perform inference
    results = model.predict(image_path)
    end_time = time.time()
    latency = end_time - start_time

    detected_objects = []
    for result in results:
        for box, mask, cls, conf in zip(result.boxes.xyxy, result.masks.data, result.boxes.cls, result.boxes.conf):
            class_idx = int(cls.item()) + 1
            if 0 <= class_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                detected_objects.append({
                    "label": COCO_INSTANCE_CATEGORY_NAMES[class_idx],
                    "confidence": float(conf),
                    "mask": mask.tolist(),
                    "box": box.tolist(),
                })

    overall_accuracy = (
        sum(obj["confidence"] for obj in detected_objects) / len(detected_objects)
        if detected_objects else 0
    )

    power_watts = get_power_usage_nvidia()
    energy_required = calculate_energy(power_watts, latency)
    image_size_bits = get_image_size_in_bits(image_path)
    throughput = (image_size_bits / latency) / 1000 if latency > 0 else 0

    return detected_objects, latency, overall_accuracy, throughput, energy_required, power_watts


# Run small segmentation
def run_small_segmentation(image_path, model_name="Unet", encoder_name="resnet34", ground_truth_path=None):
    model = load_segmentation_model_sm(model_name=model_name)
    model.eval()

    image_tensor = preprocess_image_sm(image_path)
    start_time = time.time()

    with torch.no_grad():
        predictions = model(image_tensor)

    latency = time.time() - start_time
    image_shape = Image.open(image_path).size[::-1]

    pred_mask, accuracy = process_segmentation_predictions_sm(predictions, image_shape, ground_truth_path)

    power_watts = get_power_usage_nvidia()
    energy_required = power_watts * latency
    throughput = get_image_size_in_bits(image_path) / latency if latency > 0 else 0
    # print(f"Accuracy {accuracy}")

    # visualize_segmentation(image_path, pred_mask)

    return pred_mask, latency, accuracy, throughput, energy_required, power_watts


# Main segmentation function
def run_segmentation(image_path, model_name="deeplabv3_resnet50"):
    # Load model and preprocess image
    if model_name[:2] == 'yo':
        model_ = model_name[3::]
        return run_yolo_models(image_path=image_path,model_name=model_)
    elif model_name[:2] == 'sm':
        model_ = model_name[3::]
        return run_small_segmentation(image_path=image_path,model_name=model_)
    else:
        model = load_segmentation_model(model_name)
    image_tensor = preprocess_image(image_path)
    start_time = time.time()

    # Run the model
    with torch.no_grad():
        predictions = model(image_tensor)

    end_time = time.time()
    latency = calculate_latency(start_time, end_time)

    # Extract image dimensions
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image_shape = image.size[::-1]  # Convert to (height, width)

    # Process predictions
    detected_objects, accuracy = process_segmentation_predictions(predictions, image_shape)
    # print(f"Accuracy {accuracy}")
    
    # Get power consumption
    power_watts = get_power_usage_nvidia()

    # Calculate energy consumption in joules
    energy_required = calculate_energy(power_watts, latency)

    # Calculate throughput
    image_size_bits = get_image_size_in_bits(image_path)
    throughput = (image_size_bits / latency) / 1000 if latency > 0 else 0
    
    return detected_objects, latency, accuracy, throughput, energy_required, power_watts



# Example usage
if __name__ == "__main__":
    image_path = "ppc-0603.jpeg"  # Replace with your image path or URL
    i = 1
    for model_name in MODEL_ENCODER_MAP:
        results, latency, overall_accuracy, throughput, energy_required, power_watts = run_small_segmentation(image_path, model_name)
        # print(f"Detected objects: {results}")
        print(f"Model {i} ######")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Accuracy: {overall_accuracy:.2f}")
        print(f"throughput: {throughput:.2f}")
        print(f"energy_required: {energy_required:.2f}")
        print(f"power_watts: {power_watts:.2f}")
        i+=1

# if __name__ == "__main__":
#     image_path = "ppc-0603.jpeg"  # Replace with your image path or URL
#     model_name = 'sam_h'
#     results, latency, overall_accuracy, throughput, energy_required, power_watts = run_yolo_models(image_path, model_name)
#     # print(f"Detected objects: {results}")
#     print(f"Latency: {latency:.2f} seconds")
#     print(f"Accuracy: {overall_accuracy:.2f}")
#     print(f"throughput: {throughput:.2f}")
#     print(f"energy_required: {energy_required:.2f}")
#     print(f"power_watts: {power_watts:.2f}")
    
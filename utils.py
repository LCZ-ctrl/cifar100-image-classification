import torch
import random
import numpy as np

import config
from models.squeezenet import SqueezeNet
from models.vgg16 import VGG16
from models.googlenet import GoogLeNet
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from models.densenet121 import DenseNet121
from models.densenet169 import DenseNet169
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from models.resnext50 import ResNeXt50
from models.shufflenetv1 import ShuffleNetV1
from models.shufflenetv2 import ShuffleNetV2
from models.wrn_40_4 import WideResNet_40_4
from models.vision_transformer import VisionTransformer


def calculate_correct(outputs, labels):
    return (outputs.argmax(1) == labels).sum().item()


def calculate_topk_correct(outputs, labels, k=1):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return correct.sum().item()


def build_model(model_name):
    if model_name == 'squeezenet':
        model = SqueezeNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'vgg16':
        model = VGG16(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'googlenet':
        model = GoogLeNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'mobilenetv3':
        model = MobileNetV3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'densenet121':
        model = DenseNet121(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'densenet169':
        model = DenseNet169(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'resnet34':
        model = ResNet34(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'resnext50':
        model = ResNeXt50(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'shufflenetv1':
        model = ShuffleNetV1(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'shufflenetv2':
        model = ShuffleNetV2(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'wrn_40_4':
        model = WideResNet_40_4(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif model_name == 'vision_transformer':
        model = VisionTransformer(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    return model

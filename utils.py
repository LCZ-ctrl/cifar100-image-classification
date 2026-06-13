import config
from models.squeezenet import SqueezeNet
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.googlenet import GoogLeNet
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3
from models.densenet121 import DenseNet121
from models.densenet169 import DenseNet169
from models.resnet18 import ResNet18
from models.resnet34 import ResNet34
from models.resnet50 import ResNet50
from models.resnet101 import ResNet101
from models.resnext50 import ResNeXt50
from models.resnext101 import ResNeXt101
from models.shufflenetv1 import ShuffleNetV1
from models.shufflenetv2 import ShuffleNetV2
from models.wrn_40_4 import WideResNet_40_4
from models.wrn_28_10 import WideResNet_28_10
from models.vision_transformer import VisionTransformer


def calculate_correct(outputs, labels):
    return (outputs.argmax(1) == labels).sum().item()


def calculate_topk_correct(outputs, labels, k=1):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return correct.sum().item()


MODEL_MAP = {
    'squeezenet': SqueezeNet,
    'vgg16': VGG16,
    'vgg19': VGG19,
    'googlenet': GoogLeNet,
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnext50': ResNeXt50,
    'resnext101': ResNeXt101,
    'shufflenetv1': ShuffleNetV1,
    'shufflenetv2': ShuffleNetV2,
    'wrn_40_4': WideResNet_40_4,
    'wrn_28_10': WideResNet_28_10,
    'vision_transformer': VisionTransformer
}


def build_model(model_name):
    model = MODEL_MAP.get(model_name.lower())

    if model is None:
        raise ValueError(f"Model {model_name} not found!\nValid models: {list(MODEL_MAP.keys())}")

    return model(num_classes=config.NUM_CLASSES).to(config.DEVICE)

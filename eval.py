import torch
from tqdm import tqdm
from pathlib import Path

from dataset import get_test_loader
import config
from utils import build_model, calculate_topk_correct


def evaluate_test_set(model_name='vgg16'):
    model = build_model(model_name)

    checkpoint_path = Path(config.save_folder) / model_name / f'best_model_{model_name}.pth'
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = get_test_loader()
    test_dataset = test_loader.dataset

    test_correct1 = 0.0
    test_correct5 = 0.0
    tqdm.write(f"🧠 model: {model_name}")
    print(f"🔍 Evaluating on {len(test_dataset)} test images...")

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            test_correct1 += calculate_topk_correct(outputs, labels, k=1)
            test_correct5 += calculate_topk_correct(outputs, labels, k=5)

    top1_acc = test_correct1 / len(test_dataset)
    top5_acc = test_correct5 / len(test_dataset)
    tqdm.write(f"top1 acc: {top1_acc:.4f}")
    tqdm.write(f"top5 acc: {top5_acc:.4f}")


if __name__ == "__main__":
    evaluate_test_set('resnet34')

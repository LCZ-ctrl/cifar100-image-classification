import torch

IMG_SIZE = 32
BATCH_SIZE = 128
NUM_EPOCHS = 150
LEARNING_RATE = 0.1
NUM_CLASSES = 100
NUM_WORKERS = 4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
save_folder = 'weights'

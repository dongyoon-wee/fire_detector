import torch.nn as nn
import torch
import time
import os
import copy
import argparse
from torchvision import datasets, models
from torch.utils.data import Dataset
from PIL import Image

from transform import data_transforms


parser = argparse.ArgumentParser(description='Arguments for inference')
parser.add_argument('--infer_data_dir', default='', help='data folder path for inference')
parser.add_argument('--infer_image_path', default='', help='image path for inference')
parser.add_argument('--model_path', default='', help='model path to infer')

args = parser.parse_args()


class ImgDataset(Dataset):

    def __init__(self, data_dir, transform):
        self.file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        return img_transformed


def infer_model(dataloader, model):
    since = time.time()

    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        print(preds)

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def infer_image(image, model, device):
    input = data_transforms['val'](image)
    input = input.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(input.unsqueeze(0))
        _, preds = torch.max(outputs, 1)
    return preds


if __name__=='__main__':

    # arguments
    infer_data_dir = args.infer_data_dir
    infer_image_path = args.infer_image_path
    model_path = args.model_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    assert os.path.isfile(model_path)

    model_ft.load_state_dict(torch.load(model_path))
    model_ft = model_ft.to(device)

    if os.path.isdir(infer_data_dir):
        print("infer from {}".format(infer_data_dir))
        image_dataset = ImgDataset(infer_data_dir, data_transforms['val'])
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

        infer_model(dataloader, model_ft)

    elif os.path.isfile(infer_image_path):
        preds = infer_image(Image.open(infer_image_path), model_ft, device)

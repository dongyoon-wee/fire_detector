import torch.nn as nn
import torch
import time
import copy
from torchvision import datasets, models

from transform import data_transforms


def infer_model(dataloader, model):
    since = time.time()

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        print(preds)

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__=='__main__':

    #######################################
    # things to edit
    infer_data_dir = ''
    base_lr = 0.001
    momentum = 0.9
    decay_step = 7
    decay_gamma = 0.1
    num_epochs = 25
    model_path = ''

    #######################################

    image_dataset = datasets.ImageFolder(infer_data_dir, data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft.load_state_dict(torch.load(model_path))
    model_ft = model_ft.to(device)

    model_ft = infer_model(dataloader, model_ft)

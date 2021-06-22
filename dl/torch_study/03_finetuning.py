"""
Finetuning torchvision models

Ref:
    1. https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def train_model(model: nn.Module,
                dataloaders,
                criterion,
                optimizer: optim.Optimizer,
                num_epochs=25,
                is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward, tracking history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # get model outputs and calculate loss
                    # special case for inception because in training it has an auxiliary output. In train mode we
                    # calculate the loss by summing the final output and the auxiliary output, but in testing we only
                    # consider the final output
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, val_acc_history


def set_parameter_requires_grad(model: nn.Module, feature_extract: bool):
    """
    Setting other parameters' require_grad to false for feature extracting
    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize these variable which will be set in this if statement, each of these variables is model specific
    """
    model_ft = None
    input_size = 0

    if model_name == 'resnet':
        # Resnet18
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, num_classes)
        input_size = 224
    elif model_name == 'alexnet':
        # Alexnet
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_features, num_classes)
        input_size = 224
    elif model_name == 'vgg':
        # VGG11_bn
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_features, num_classes)
        input_size = 224
    elif model_name == 'squeezenet':
        # Squeezenet
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = model_ft.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == 'densenet':
        # Densenet
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_features, num_classes)
        input_size = 224
    elif model_name == 'inception':
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # handle the auxiliary net
        num_features = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_features, num_classes)
        # handle the primary net
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, num_classes)
        input_size = 299
    else:
        print('Invalid model name, exiting...')
        exit()

    return model_ft, input_size


if __name__ == '__main__':
    print(f'PyTorch Version: {torch.__version__}')
    print(f'Torchvision Version: {torchvision.__version__}')

    # define some parameters
    data_dir = 'data/hymenoptera_data'  # top level data directory
    model_name = 'inception'  # model name
    num_classes = 2  # number of classes in dataset
    batch_size = 8  # batch size for traning
    num_epochs = 15  # number of epochs to train for
    # flag for feature extracting, we fine tune the whole model for False, and only update the reshaped layer params for True
    feature_extract = True

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)

    # data augmentation and normalization for training
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }
    print('Initializing Datasets and Dataloaders...')
    # create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    # detect if we have a GPU available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # send the model to device
    model_ft = model_ft.to(device)

    # gather the parameters to be optimized/updated in this run
    params_to_update = model_ft.parameters()
    print('Parameters to Learn:')
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print(f'\t{name}')
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print(f'\t{name}')

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # setup the loss function
    criterion = nn.CrossEntropyLoss()

    # train and evaluate
    model_ft, hist = train_model(model_ft,
                                 dataloaders_dict,
                                 criterion,
                                 optimizer_ft,
                                 num_epochs=num_epochs,
                                 is_inception=(model_name == 'inception'))

    # initialize the non-pretrained version of the model used for this run
    scratch_model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist = train_model(scratch_model,
                                  dataloaders_dict,
                                  scratch_criterion,
                                  scratch_optimizer,
                                  num_epochs=num_epochs,
                                  is_inception=(model_name == 'inception'))

    # plot the training curves of validation accuracy vs. number of training epochs for the transfer learning method
    # and the model trained from scratch
    ohist = [h.cpu().numpy() for h in hist]
    shist = [h.cpu().numpy() for h in scratch_hist]
    plt.title('Validation Accuracy vs. Number of Training Epochs')
    plt.xlabel('Training Epochs')
    plt.ylabel('Validation Accuracy')
    plt.plot(range(1, num_epochs + 1), ohist, label='Pretrained')
    plt.plot(range(1, num_epochs + 1), shist, label='Scratch')
    plt.ylim(0, 1)
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()

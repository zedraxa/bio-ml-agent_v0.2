import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from collections import defaultdict
import json

class MedicalCNN(nn.Module):
    def __init__(self, num_classes, architecture='resnet18', pretrained=True):
        super(MedicalCNN, self).__init__()
        if architecture == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif architecture == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif architecture == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif architecture == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif architecture == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x):
        return self.model(x)

def quick_train_cnn(data_dir, preset="brain_mri", architecture="resnet18", epochs=25, output_dir="results/"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data transformations based on preset
    if preset == "brain_mri":
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True if x == 'train' else False,
                                                 num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    model = MedicalCNN(num_classes=num_classes, architecture=architecture, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics_history = defaultdict(list)

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            metrics_history[f'{phase}_loss'].append(epoch_loss)
            metrics_history[f'{phase}_accuracy'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    # Save metrics history
    with open(os.path.join(output_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f)
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.plot(metrics_history['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics_history['val_accuracy'], label='Validation Accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plots', 'learning_curve.png'))
    plt.close()

    final_metrics = {
        'train_accuracy': metrics_history['train_accuracy'][-1],
        'val_accuracy': metrics_history['val_accuracy'][-1],
        'train_loss': metrics_history['train_loss'][-1],
        'val_loss': metrics_history['val_loss'][-1],
    }
    return final_metrics

def compare_architectures(data_dir, preset="brain_mri", architectures=None, epochs_per_trial=10, output_dir="results/"):
    if architectures is None:
        architectures = ["resnet18", "densenet121", "efficientnet_b0"]
    
    results = {}
    for arch in architectures:
        print(f"\nTraining with {arch} architecture...")
        metrics = quick_train_cnn(data_dir, preset, arch, epochs_per_trial, output_dir=os.path.join(output_dir, arch))
        results[arch] = metrics
        print(f"Results for {arch}: {metrics}")
    
    # Save comparison results
    with open(os.path.join(output_dir, "architecture_comparison.json"), "w") as f:
        json.dump(results, f)
    
    return results
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import copy
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Dummy for utils.visualize and xai_engine if they are not provided, or to ensure no import error
try:
    from utils.visualize import MLVisualizer
    from xai_engine import XAIEngine
except ImportError:
    print("Warning: MLVisualizer or XAIEngine not found. Using dummy classes.")
    class MLVisualizer:
        def __init__(self, output_dir="results/plots"):
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names=None, df=None, class_names=None):
            print(f"Plotting all visualizations (dummy): {self.output_dir}")
            if class_names is None:
                if y_test is not None:
                    class_names = [str(i) for i in sorted(np.unique(y_test))]
                else:
                    class_names = ["0", "1"] # Default
            # Dummy confusion matrix
            fig, ax = plt.subplots()
            sns.heatmap([[0.5, 0.5], [0.5, 0.5]], annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title("Dummy Confusion Matrix")
            plt.savefig(os.path.join(self.output_dir, "dummy_confusion_matrix.png"))
            plt.close()
            # Dummy ROC curve
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title("Dummy ROC Curve")
            plt.savefig(os.path.join(self.output_dir, "dummy_roc_curve.png"))
            plt.close()

    class XAIEngine:
        def __init__(self, model, X_train, feature_names=None, task_type="classification"):
            self.model = model
            self.X_train = X_train
            self.feature_names = feature_names
            self.task_type = task_type
            print("XAI Engine initialized with dummy model/data.")
        def generate_shap_summary(self, X_test, output_dir="results/plots", max_display=10):
            print(f"Generating SHAP summary (dummy): {output_dir}")
            # Simulate a SHAP plot
            fig, ax = plt.subplots()
            ax.barh(["Feature A", "Feature B"], [0.6, 0.4])
            ax.set_title("Dummy SHAP Summary")
            plt.savefig(os.path.join(output_dir, "dummy_shap_summary.png"))
            plt.close()
        def explain_instance_lime(self, instance, output_dir="results/plots"):
            print(f"Explaining instance with LIME (dummy): {output_dir}")
            # Simulate a LIME plot
            fig, ax = plt.subplots()
            ax.barh(["Region 1", "Region 2"], [0.7, 0.3])
            ax.set_title("Dummy LIME Explanation")
            plt.savefig(os.path.join(output_dir, "dummy_lime_explanation.png"))
            plt.close()


class MedicalCNN:
    def __init__(self, architecture="resnet18", num_classes=2, pretrained=True, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if architecture == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif architecture == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif architecture == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        elif architecture == "densenet121":
            self.model = models.densenet121(pretrained=pretrained)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
        elif architecture == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train_model(self, dataloaders, num_epochs=25, output_dir="results/"):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}'):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        self.model.load_state_dict(best_model_wts)
        
        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f)

        return self.model, history

    def evaluate_model(self, dataloader, class_names):
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # For ROC AUC, handle binary vs multi-class
        if len(class_names) == 2:
            roc_auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1], average='weighted')
        else:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        return metrics, all_labels, all_preds, all_probs


def quick_train_cnn(data_dir, preset, architecture="resnet18", epochs=25, batch_size=32, output_dir="results/"):
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val', 'test']
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")

    if num_classes < 2:
        raise ValueError("At least 2 classes are required for classification.")

    # Initialize model
    cnn_model = MedicalCNN(architecture=architecture, num_classes=num_classes)

    # Train model
    best_model, history = cnn_model.train_model(
        {'train': dataloaders['train'], 'val': dataloaders['val']},
        num_epochs=epochs,
        output_dir=output_dir
    )

    # Save the best model
    model_save_path = os.path.join(output_dir, "best_model.pkl")
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    # Evaluate on test set
    test_metrics, all_labels, all_preds, all_probs = cnn_model.evaluate_model(dataloaders['test'], class_names)
    print("\nTest Set Metrics:")
    print(test_metrics)

    # Save metrics
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Plot visualizations
    viz = MLVisualizer(output_dir=os.path.join(output_dir, "plots"))
    
    # Need to convert all_labels and all_preds to numpy arrays if they aren't already
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    
    # Since MLVisualizer is generic, it might need features, but for image classification, we just pass None
    # Assuming MLVisualizer's plot_all is adapted for image classification or can handle None for features
    viz.plot_all(best_model, None, None, all_labels_np, all_preds_np, class_names=class_names, is_image_classification=True)

    # XAI
    # For XAI on images, we typically use Grad-CAM or similar. SHAP/LIME are harder on raw images.
    # For now, we'll use the dummy XAIEngine if it's not a real one.
    if 'XAIEngine' in globals() and XAIEngine.__name__ != 'XAIEngine': # Check if it's the real class
        # XAI for image models often involves Grad-CAM or similar techniques
        # SHAP/LIME are more for tabular data or image features, not raw pixels directly in this way
        # For simplicity with the current XAIEngine, we might need a workaround or a specialized image XAI tool
        # For now, if the real XAIEngine needs X_train, we will pass a dummy or sample
        # Since this is a placeholder, we'll use a single dummy image from the test set for LIME
        if len(image_datasets['test']) > 0:
            sample_image, _ = image_datasets['test'][0]
            # SHAP/LIME for CNNs usually requires a custom explainer (e.g., DeepExplainer for SHAP)
            # The current XAIEngine is generic and expects X_train as a feature matrix
            # We will generate dummy SHAP/LIME plots for now.
            print("Warning: Real XAIEngine requires image-specific integration. Generating dummy plots.")
            xai = XAIEngine(best_model, None, feature_names=None, task_type="classification")
            xai.generate_shap_summary(None, output_dir=os.path.join(output_dir, "plots")) # None for X_test as it's image based
            xai.explain_instance_lime(None, output_dir=os.path.join(output_dir, "plots")) # None for instance as it's image based
        else:
            print("No test images to perform XAI on (dummy).")
    else:
        print("Using dummy XAIEngine for XAI. Real XAI for image classification needs specialized tools like Grad-CAM.")
        xai_dummy = XAIEngine(best_model, None, feature_names=None, task_type="classification")
        xai_dummy.generate_shap_summary(None, output_dir=os.path.join(output_dir, "plots"))
        xai_dummy.explain_instance_lime(None, output_dir=os.path.join(output_dir, "plots"))


    # Return test metrics
    return test_metrics

def compare_architectures(data_dir, preset, architectures=["resnet18"], epochs=25, batch_size=32, output_dir="results/"):
    results = {}
    for arch in architectures:
        print(f"\n--- Training with {arch} ---")
        metrics = quick_train_cnn(data_dir, preset, architecture=arch, epochs=epochs, batch_size=batch_size, output_dir=output_dir)
        results[arch] = metrics
    
    # Save comparison results
    with open(os.path.join(output_dir, "architecture_comparison.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- Architecture Comparison Results ---")
    for arch, met in results.items():
        print(f"Architecture: {arch}, Metrics: {met}")
    
    return results

# Dummy function for model_compare. This agent is not using it for deep learning.
def compare_models(*args, **kwargs):
    print("Dummy compare_models called. This is for traditional ML, not deep learning.")
    return None, None

# Dummy function for hyperparameter_optimizer.
def optimize_model(*args, **kwargs):
    print("Dummy optimize_model called. This is for traditional ML, not deep learning.")
    return None, None
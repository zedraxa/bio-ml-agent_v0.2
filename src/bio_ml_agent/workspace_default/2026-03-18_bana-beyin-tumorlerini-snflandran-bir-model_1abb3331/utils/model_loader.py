import torch
import os
from deep_learning import MedicalCNN # Import your model class

def load_and_predict(model_path, X_new, num_classes=4, architecture='resnet18', device=None):
    """
    Loads a trained PyTorch model and makes predictions on new data.

    Args:
        model_path (str): Path to the saved model (.pth file).
        X_new (torch.Tensor or numpy.ndarray): New data to predict on.
                                              Assumes X_new is already preprocessed.
        num_classes (int): Number of output classes for the model.
        architecture (str): Architecture of the model (e.g., 'resnet18').
        device (torch.device, optional): Device to load the model on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        torch.Tensor: Predicted probabilities for each class.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with the same architecture and number of classes as trained
    model = MedicalCNN(num_classes=num_classes, architecture=architecture, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode
    model.to(device)

    # Ensure X_new is a torch.Tensor and on the correct device
    if isinstance(X_new, np.ndarray):
        X_new = torch.from_numpy(X_new).float()
    X_new = X_new.to(device)

    with torch.no_grad():
        outputs = model(X_new)
        probabilities = torch.softmax(outputs, dim=1)

    return probabilities
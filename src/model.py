import torch.nn as nn
from torchvision import models

def initialize_model(num_classes):
    print("Loading pre-trained ResNet18 architecture...")
    
    # Load a model that already knows how to detect edges, shapes, and textures
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze the early layers so we don't destroy the pre-trained knowledge
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Replacing the final classification head for {num_classes} classes...")
    # Replace the final fully connected layer (which originally had 1000 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

import os
from data_setup import create_dataloaders
from model import initialize_model
from train import train_vision_model

def run_vision_pipeline():
    # Expecting a structure like: data/train/glass/, data/train/plastic/, etc.
    data_dir = "../data"
    
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print(f"Error: Please set up your image directory structure at {data_dir}")
        print("Required structure: data/train/[class_name] and data/val/[class_name]")
        return

    try:
        # 1. Load Data
        dataloaders, class_names = create_dataloaders(data_dir)
        num_classes = len(class_names)
        
        # 2. Initialize Transfer Learning Model
        model = initialize_model(num_classes)
        
        # 3. Train the Model
        trained_model = train_vision_model(model, dataloaders, num_epochs=5)
        
        print("\nOptical Sorting Pipeline executed successfully.")

    except Exception as e:
        print(f"\nPipeline failed: {e}")

if __name__ == "__main__":
    run_vision_pipeline()

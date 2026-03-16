# Recyclable Material Classification (Computer Vision)

## 📌 Project Overview
In modern material recovery facilities (MRFs), the purity of recycled bales determines their market value. Manual sorting is inefficient and hazardous, while traditional mechanical sorters struggle to differentiate between complex materials (e.g., clear PET plastic vs. glass).

This project implements an automated **Optical Sorting Computer Vision Pipeline**. It utilizes **PyTorch** and Transfer Learning (ResNet18) to build an image classification neural network capable of identifying and categorizing recyclable materials on a simulated high-speed conveyor belt.

**Business Value:** Automates quality control in waste management, minimizes stream contamination, and increases the raw material yield for circular economy divisions (like PreZero).

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Architecture:** Convolutional Neural Networks (CNNs), ResNet18
* **Technique:** Transfer Learning & Data Augmentation
* **Language:** Python 3.x

## 🏗️ Architecture & Workflow
1. **Data Augmentation:** Utilizes `torchvision.transforms` to dynamically alter training images (random cropping, flipping) to prevent model overfitting and simulate chaotic conveyor belt conditions.
2. **Transfer Learning:** Ingests a pre-trained ResNet18 model, freezing the foundational feature-extraction layers to vastly reduce required compute power and training time.
3. **Custom Classification Head:** Replaces the final fully connected layer to map specifically to recycling classes (e.g., Cardboard, Glass, Metal, Paper, Plastic).
4. **GPU Optimization:** Implements dynamic device allocation to train the network efficiently on CUDA-enabled GPUs, tracking Epoch Loss and Accuracy to ensure high-confidence sorting.

## 📂 Project Structure
```text
├── data/                   # Image directory (Excluded via .gitignore)
│   ├── train/              # Training images separated by class folders
│   └── val/                # Validation images separated by class folders
├── src/                    
│   ├── data_setup.py       # Image transforms and DataLoader creation
│   ├── model.py            # ResNet18 Transfer Learning architecture
│   ├── train.py            # Backpropagation and accuracy tracking
│   └── main.py             # Pipeline orchestrator
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation

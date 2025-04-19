# DA6401-Assignment-2-PartB
This documentation explains a PyTorch-based approach for fine-tuning a pre-trained ResNet50 model on a subset of the iNaturalist dataset for image classification, with performance tracking using Weights & Biases (wandb).

## Overview
This project demonstrates transfer learning by taking a ResNet50 model (pre-trained on ImageNet) and adapting it to classify natural world images from the iNaturalist dataset. The implementation strategically freezes early layers of the network while training only the later layers, which is an effective technique for transfer learning when working with limited domain-specific data.

### Project Structure
```
DA6401-Assignment-2-PartA/
    ├── .gitignore              # Includes dataset and wandb folder
    ├── cnn_training.ipynb      # Jupyter notebook - contains all of the code necessary for Part A of the assignment
    ├── README.md               # Project documentation
    └── resnet50_finetuned.pth  # Fine-tuned model with the best val accuracy achieved with wandb run
```

### Requirements
To run this notebook, you'll need the following libraries:
- PyTorch (torch)
- Torchvision (torchvision)
- Weights & Biases (wandb)

as well as the iNaturalist_12K dataset, structured as follows:
```
nature_12K/
└── inaturalist_12K/
    ├── train/
    │   └── [class folders]/
    └── val/
        └── [class folders]/
```

## Setup

- Install required packages:
    ```
    pip install torch torchvision numpy scikit-learn matplotlib pillow wandb
    ```
    If your device has an nvidia gpu supporting CUDA, install the required drivers, CUDA toolkit and Microsoft Visual Studio 2022. Then install the pytorch packages with:
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```
    **Note:** Ensure that the CUDA toolkit installed is supported by PyTorch. The above installation is compatible with CUDA version 12.6.
- Go to your Weights & Biases account (https://wandb.ai) and authenticate:
    ```
    import wandb
    wandb.login()
    ```

## Sections of the Notebook
1. ***Library Imports:*** Imports necessary PyTorch libraries, data handling tools, and wandb for experiment tracking.
2. ***Weights & Biases Initialization:*** Sets up wandb project to track model performance and experiment metrics.
3. ***Data Transformations:*** Defines two sets of image transformations:
    - Train transformations: Includes data augmentation techniques like random crops and horizontal flips
    - Test transformations: Uses standardized resizing and center cropping
    Both use normalization with ImageNet mean and standard deviation values
4. ***Dataset Loading:*** Loads the iNaturalist dataset using PyTorch's ImageFolder to automatically handle the class structure.
5. ***Dataset Splitting:*** Creates a validation set by splitting 20% of the training data, keeping 80% for actual training.
6. ***Data Loaders:*** Configures PyTorch DataLoaders with batch size 32 for efficient training, validation, and testing.
7. ***Model Loading:*** Loads a pre-trained ResNet50 model with ImageNet weights as the base model.
8. ***Layer Freezing:*** Freezes the early layers (up to layer2) to preserve learned features while allowing adaptation in later layers.
9. ***Final Layer Replacement:*** Replaces the final fully connected layer to accommodate the 10 classes in the iNaturalist subset.
10. ***Device Selection:*** Automatically selects GPU if available, otherwise falls back to CPU.
11. ***Loss Function and Optimizer:*** Uses cross-entropy loss and Adam optimizer with differential learning rates:
    - Later convolutional layers (3-4): Learning rate of 1e-4
    - Final classification layer: Higher learning rate of 1e-3
12. ***Training Loop:*** Implements a 10-epoch training process with:
    - Forward/backward passes for parameter updates
    - Tracking of training and validation metrics
    - Real-time logging to wandb dashboard
    - Progress updates printed to console
13. ***Testing:*** Evaluates the final model performance on the separate test dataset.
14. ***Model Saving:*** Saves the trained model weights and finalizes the wandb logging session.

## Performance Results
The model achieves the following metrics after 10 epochs:
- Final training accuracy: 86.16%
- Final validation accuracy: 74.85%
- Test accuracy: 81.20%

## Key Design Choices
- Transfer Learning Strategy: By freezing early layers and only fine-tuning later layers, the model leverages ImageNet pre-trained features while adapting to the specific task.
- Learning Rate Differentiation: Different learning rates for different layer groups allow for appropriate parameter adjustments:
    - Later convolutional layers (3-4): 1e-4
    - Final classification layer: 1e-3
- Data Augmentation: Training images undergo random crops and flips to improve generalization.
- Training/Validation Split: Using 80% of the training data for actual training and 20% for validation helps monitor overfitting.

## Inference
To use the trained model for inference:
- Load the saved model weights
- Process input images with the test transformations
- Run the model in evaluation mode
- Interpret the output predictions
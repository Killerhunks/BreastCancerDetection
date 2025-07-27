# Breast Cancer Detection

This project provides a comprehensive pipeline for breast cancer detection using deep learning and transfer learning techniques. The workflow is implemented in the `BreastCanerDetection.ipynb` notebook and leverages several state-of-the-art models for image classification.

## Features

- **Dataset Preparation:** Automated splitting of the BreaKHis dataset into train, validation, and test sets.
- **Data Augmentation:** Extensive augmentation strategies to improve model generalization.
- **Model Architectures:** 
  - Custom CNN
  - Transfer learning with ResNet50, MobileNet, VGG16
  - EfficientNetB0 with fine-tuning
  - Vision Transformer (ViT) integration (experimental)
- **Imbalanced Data Handling:** Class weighting and custom focal loss.
- **Training Utilities:** Early stopping, learning rate scheduling, and training history visualization.
- **Evaluation Metrics:** Accuracy, ROC-AUC, F1 Score, and detailed classification reports.
- **Model Comparison:** Visualization of accuracy and loss across different models.

## Usage

1. **Dataset Setup:**  

   Place the BreaKHis dataset in the specified directory structure:
   dataset/
   ├── train/ 
   ├── val/ 
   └── test/

The notebook includes code to automate this split.

2. **Run the Notebook:**  
Open `BreastCanerDetection.ipynb` in VS Code and execute the cells sequentially.

3. **Model Training:**  
- Train custom CNN and transfer learning models.
- Fine-tune EfficientNetB0 with focal loss for class imbalance.
- Optionally, experiment with Vision Transformer models.

4. **Evaluation & Visualization:**  
- View classification reports and metrics.
- Compare model performance using provided plots.

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- matplotlib
- transformers (for ViT, optional)

Install dependencies:
```bash
pip install tensorflow scikit-learn matplotlib transformers
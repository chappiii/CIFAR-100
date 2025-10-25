# CIFAR-100 Deep Learning Model Benchmarking

## Overview
This project explores deep learning for image classification on the CIFAR-100 dataset. I designed and implemented a custom convolutional neural network (CNN) and benchmarked its performance against leading architectures: ResNet-50, VGG-19, DenseNet-121, and EfficientNet-B0. The goal was to achieve strong accuracy and generalization with efficient training, and to analyze the trade-offs between custom and standard models.

## Dataset & Preprocessing
- **Dataset:** CIFAR-100, 60,000 color images (32x32 pixels) across 100 fine-grained categories.
- **Preprocessing:**
  - One-hot encoding of labels for multi-class classification.
  - Images normalized to [0, 1].
  - For ResNet, DenseNet, and EfficientNet, images resized to 128x128; for the custom CNN and VGG-19, original 32x32 size retained.
  - Training set shuffled and split (90% train, 10% validation).
  - Mini-batch training (batch size = 64) for efficient memory usage.

## Custom CNN Architecture
- Four convolutional blocks with increasing filter depth (32 → 256), each followed by batch normalization, ReLU activation, and dropout for regularization.
- Final classification head: Flatten → Dense(512) → BatchNorm → ReLU → Dropout → Dense(100, softmax).
- Total parameters: ~3.3M.
- Regularization: Dropout rates increase with depth (0.25–0.5) to prevent overfitting.
- Optimizer: Adam; Loss: categorical crossentropy; Early stopping used for best validation performance.

## Training & Evaluation
- The custom CNN achieved:
  - **Test Accuracy:** 56.2%
  - **Macro F1-score:** 55.8%
  - **Training Time:** 192 seconds (25 epochs)
- Training and validation curves show strong convergence and minimal overfitting.
- Visualized convolutional filters before and after training to interpret learned features.

## Model Comparison
| Model           | Train Acc | Val Acc | Test Acc | Time (s) | Epochs | Notes                  |
|-----------------|-----------|---------|----------|----------|--------|------------------------|
| Custom CNN      | 62.5%     | 54.9%   | 56.2%    | 192      | 25     | Best balance, robust   |
| DenseNet-121    | 88.1%     | 57.8%   | 56.5%    | 2244     | 13     | Highest accuracy, slow |
| ResNet-50       | 86.6%     | 44.5%   | 38.8%    | 1630     | 11     | Overfitting observed   |
| EfficientNet-B0 | 87.1%     | 39.2%   | 51.5%    | 726      | 13     | Overfitting observed   |
| VGG-19          | 18.4%     | 19.0%   | 18.6%    | 1180     | 23     | Underperformed         |

- The custom CNN outperformed or matched deeper models in accuracy and efficiency, especially under resource constraints.
- Advanced models showed overfitting or required much longer training times.
- DenseNet-121 achieved the highest test accuracy but at a significant computational cost.

## Key Achievements
- Designed a custom CNN that balances accuracy, generalization, and computational efficiency for CIFAR-100.
- Demonstrated that a well-optimized, lightweight model can outperform deeper architectures in constrained environments.
- Provided comprehensive benchmarking and analysis, including training curves, filter visualization, and confusion matrix.
- Gained hands-on experience with TensorFlow/Keras, model evaluation, and deep learning best practices.

## Conclusion
This project highlights the effectiveness of custom, resource-efficient CNNs for complex image classification tasks. While DenseNet-121 achieved the highest test accuracy, it required significantly more training time and showed signs of overfitting, highlighting the practical efficiency and robustness of the custom CNN in real-world, resource-constrained scenarios. While state-of-the-art models may surpass the custom CNN with further tuning, data augmentation, and computational power, this work demonstrates that simplicity and targeted optimization can yield highly competitive results in moderately complex classification problems, making the custom CNN a practical and robust solution.

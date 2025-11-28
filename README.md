# Road Segmentation with U-Net

A deep learning project for semantic segmentation of roads from satellite imagery using a U-Net architecture. Trained on the DeepGlobe Road Extraction Dataset.

## Project Overview

This project implements a U-Net convolutional neural network to automatically identify and segment roads from satellite images. The model achieves strong performance with a validation Dice coefficient of 0.6716 and IoU of 0.5062.

## Results

| Metric               | Training | Validation |
| -------------------- | -------- | ---------- |
| **Loss**             | 0.0471   | 0.0505     |
| **Dice Coefficient** | 0.6730   | 0.6716     |
| **IoU**              | 0.5074   | 0.5062     |
| **Accuracy**         | 98.13%   | 98.06%     |

## Model Architecture

The implementation uses a classic U-Net architecture with:

- **Encoder**: 4 blocks with progressively increasing filters (32 → 64 → 128 → 256)
- **Bridge**: 512 filters
- **Decoder**: 4 blocks with skip connections from encoder
- **Features**:
  - Batch normalization after each convolution
  - Dropout (0.3) for regularization
  - Conv2DTranspose for upsampling
  - Sigmoid activation for binary output

**Total Parameters**: 7.77M (7.76M trainable)

## Dataset

**DeepGlobe Road Extraction Dataset**

- Training samples: 5,603
- Validation samples: 623
- Image size: 256×256×3
- Binary masks for road segmentation

## Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **scikit-learn**: Train/test splitting

## Training Configuration

```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
OPTIMIZER = Adam
LOSS = Binary Crossentropy
```

### Callbacks

- **ModelCheckpoint**: Saves best model based on validation Dice coefficient
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **EarlyStopping**: Stops training if validation Dice doesn't improve for 10 epochs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/road-segmentation.git
cd road-segmentation

# Install dependencies
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

### Training

```python
# Run the training notebook or script
python train.py
```

### Inference

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('final_unet_model.keras',
                   custom_objects={'dice_coef': dice_coef, 'iou_coef': iou_coef})

# Make predictions
predictions = model.predict(test_images)
predictions = (predictions > 0.5).astype(np.float32)
```

## Training Progress

The model was trained for 50 epochs with learning rate reduction:

- Initial LR: 0.001
- Reduced to: 0.0005 (epoch 19), 0.00025 (epoch 27), 0.000125 (epoch 38), 0.0000625 (epoch 44)

Training time: ~120 seconds per epoch on dual Tesla T4 GPUs

## Output Files

- `road_segmentation.keras`: Best model checkpoint
- `final_unet_model.keras`: Final trained model
- `training_history.json`: Complete training history
- `final_metrics.txt`: Final evaluation metrics
- `iou_scores.npy`: IoU scores for all validation samples

## Custom Metrics

The project implements custom evaluation metrics:

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index for segmentation quality

## Data Pipeline

Highly optimized `tf.data` pipeline with:

- Parallel data loading with `AUTOTUNE`
- Image normalization to [0,1]
- Binary thresholding for masks
- Batch prefetching for faster training

## Visualization

The notebook includes functions for:

- Displaying sample predictions
- Plotting training curves (loss, Dice, IoU)
- IoU distribution histograms
- Side-by-side comparison of input/ground truth/prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- DeepGlobe Challenge for the dataset
- U-Net architecture by Ronneberger et al.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project was developed and trained on Kaggle with dual Tesla T4 GPUs.

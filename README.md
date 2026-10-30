# Applied AI Midterm — SRGAN-Assisted Cat vs Dog Classification

## Overview

This project investigates whether **Super-Resolution Generative Adversarial Networks (SRGAN)** can reconstruct useful high-resolution image information from very low-resolution inputs for a binary image-classification task.

The original task is **Cat vs Dog classification**. Two classifiers are compared:

- **Model A:** MobileNetV2 trained on original images resized directly to **128×128**
- **Model B:** The same MobileNetV2 classifier trained on images first downscaled to **32×32**, reconstructed to **128×128 using SRGAN**, and then classified

Using the same classifier architecture for both models keeps the comparison fair and isolates the effect of the SRGAN-based image reconstruction pipeline.

---

## Dataset Split

The full dataset contained **24,930 usable images**.

A fixed stratified split was used:

| Split | Number of Images | Percentage |
|---|---:|---:|
| Training | 17,451 | 70% |
| Testing | 7,479 | 30% |

The same split was used throughout the experiment.

The training set was approximately balanced:

- Cat: 8,730
- Dog: 8,721

A fixed random seed was used for reproducibility.

---

## Overall Pipeline

### Model A

```text
Original image
      ↓
Resize to 128×128
      ↓
MobileNetV2
      ↓
Cat / Dog
```

### Model B

```text
Original image
      ↓
Resize to 128×128
      ↓
Downscale to 32×32
      ↓
SRGAN Generator
      ↓
Reconstruct to 128×128
      ↓
MobileNetV2
      ↓
Cat / Dog
```

---

# 1. Environment Setup

The project was developed using Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Pillow, and scikit-learn.

Example environment setup:

```bash
conda create -n tf_midterm python=3.11 -y
conda activate tf_midterm

python -m pip install --upgrade pip
python -m pip install tensorflow numpy pandas matplotlib scikit-learn pillow notebook tensorboard
```

To launch Jupyter:

```bash
python -m notebook
```

---

## GPU Selection

On the Lambda server, CUDA GPU 1 was selected before importing TensorFlow:

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
```

TensorFlow may display this GPU internally as `GPU:0` because only one GPU is visible after setting `CUDA_VISIBLE_DEVICES`.

Memory growth was enabled:

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

# 2. Reproducibility Settings

A fixed seed was used:

```python
SEED = 42

import random
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
```

---

# 3. Model A — Baseline Classifier

Model A uses original images resized directly to **128×128**.

The classifier is based on **MobileNetV2 pretrained on ImageNet**.

```python
base_model_A = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)
```

The original ImageNet classification head was removed with:

```python
include_top=False
```

A custom binary-classification head was added:

```text
MobileNetV2
     ↓
GlobalAveragePooling2D
     ↓
Dropout(0.3)
     ↓
Dense(128, ReLU)
     ↓
Dense(1, Sigmoid)
```

The sigmoid output represents the probability of the positive class.

---

## Model A Transfer Learning

Initially, the MobileNetV2 base was frozen:

```python
base_model_A.trainable = False
```

The model was trained for **10 epochs** using:

```text
Optimizer: Adam
Learning rate: 0.001
Loss: Binary cross-entropy
```

---

## Model A Fine-Tuning

After transfer learning, the deeper MobileNetV2 layers were fine-tuned.

Approximately the last 30 layers were considered for unfreezing, while BatchNormalization layers remained frozen.

The fine-tuning learning rate was reduced to:

```text
1e-5
```

The model was fine-tuned for another **10 epochs**.

---

# 4. SRGAN Configuration

The SRGAN was trained to reconstruct:

```text
32×32 → 128×128
```

Configuration:

```python
SR_LR_SIZE = (32, 32)
SR_HR_SIZE = (128, 128)

SR_BATCH_SIZE = 16
SRGAN_EPOCHS = 150
GENERATOR_PRETRAIN_EPOCHS = 5
```

A batch size of 16 was used during GAN training to control GPU-memory usage.

The required SRGAN training length was **150 epochs**.

---

# 5. SRGAN Image Preprocessing

Each original training image was:

1. Loaded using Pillow
2. Converted to RGB
3. Resized to 128×128 to create the HR target
4. Downscaled to 32×32 to create the LR input
5. Normalized from `[0, 1]` to `[-1, 1]`

Pillow was used because some files in the dataset had non-standard or truncated image encodings.

Example:

```python
with Image.open(path) as img:
    img = img.convert("RGB")
    image = np.array(img, dtype=np.float32) / 255.0
```

Normalization:

```python
lr_image = (lr_image * 2.0) - 1.0
hr_image = (hr_image * 2.0) - 1.0
```

This matches the generator output activation, which uses `tanh`.

---

## SRGAN Data Augmentation

Training-only augmentation included:

```text
Horizontal flipping
Brightness variation
Contrast variation
```

These transformations were applied to the HR image before creating the LR image so that each LR/HR pair remained aligned.

---

# 6. SRGAN Generator

The generator receives:

```text
32×32×3
```

and produces:

```text
128×128×3
```

Main structure:

```text
Input 32×32×3
      ↓
9×9 Conv, 64 filters
      ↓
PReLU
      ↓
8 Residual Blocks
      ↓
Global Skip Connection
      ↓
Upsampling Block
32×32 → 64×64
      ↓
Upsampling Block
64×64 → 128×128
      ↓
9×9 Conv, 3 filters
      ↓
tanh
      ↓
Output 128×128×3
```

---

## Residual Blocks

Each residual block contains:

```text
Input
 ↓
3×3 Conv
 ↓
BatchNormalization
 ↓
PReLU
 ↓
3×3 Conv
 ↓
BatchNormalization
 ↓
Add original input
```

Eight residual blocks were used.

Residual connections help preserve information and improve gradient flow.

---

## PixelShuffle Upsampling

The generator uses `PixelShuffle(scale=2)`.

For scale 2:

```text
H × W × C
      ↓
2H × 2W × C/4
```

Example:

```text
32×32×256
      ↓ PixelShuffle
64×64×64
```

Before the second PixelShuffle, a convolution again creates 256 channels:

```text
64×64×64
      ↓ Conv2D(256)
64×64×256
      ↓ PixelShuffle
128×128×64
```

Two PixelShuffle blocks are therefore sufficient for:

```text
32 → 64 → 128
```

---

# 7. SRGAN Discriminator

The discriminator receives a 128×128 image and predicts whether it is:

- a real HR image, or
- an SRGAN-generated image

The convolutional filters increase progressively:

```text
64
64
128
128
256
256
512
512
```

while stride-2 convolutions reduce spatial resolution:

```text
128 → 64 → 32 → 16 → 8
```

LeakyReLU with slope `0.2` was used.

The final layer is:

```python
Dense(1)
```

with no sigmoid because:

```python
BinaryCrossentropy(from_logits=True)
```

is used.

---

# 8. VGG19 Perceptual Feature Extractor

A pretrained VGG19 network was used only to calculate perceptual loss.

```python
vgg19 = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)
```

Features were extracted from:

```text
block5_conv4
```

The VGG19 model was frozen:

```python
vgg_features.trainable = False
```

VGG19 is not used for cat/dog classification.

Its purpose is to compare high-level visual features between real HR images and generated SR images.

---

# 9. SRGAN Loss Functions

The discriminator uses binary cross-entropy.

The generator uses three loss components.

## Pixel Loss

```python
mean(abs(hr - sr))
```

This measures pixel-level reconstruction error.

## Perceptual Loss

VGG19 features are extracted from the HR and SR images:

```text
HR → VGG19 → HR features
SR → VGG19 → SR features
```

The feature representations are compared using mean squared error.

## Adversarial Loss

The generator attempts to make the discriminator classify generated images as real.

---

## Total Generator Loss

The final generator objective was:

```text
Generator Loss
=
Pixel Loss
+
0.001 × Perceptual Loss
+
0.001 × Adversarial Loss
```

The weights prevent the larger numerical perceptual and adversarial losses from overwhelming the pixel-reconstruction term.

---

# 10. Generator Pretraining

Before adversarial training, the generator was pretrained for **5 epochs** using mean squared reconstruction error.

Observed pretraining loss:

| Epoch | Loss |
|---:|---:|
| 1 | 0.02971 |
| 2 | 0.01600 |
| 3 | 0.01410 |
| 4 | 0.01322 |
| 5 | 0.01262 |

The decreasing loss shows that the generator learned a useful initial reconstruction before adversarial competition began.

---

# 11. SRGAN Training

The generator and discriminator were both optimized using Adam:

```text
Learning rate: 1e-4
beta_1: 0.9
```

The SRGAN was trained for:

```text
150 epochs
```

Checkpoints were saved every:

```text
5 epochs
```

The checkpoint includes:

```text
Generator state
Discriminator state
Generator optimizer state
Discriminator optimizer state
Current epoch
```

This allows training to resume after interruption.

---

## SRGAN Training Behavior

The generator loss decreased approximately from:

```text
0.1324 → 0.0950
```

The perceptual loss decreased approximately from:

```text
51.36 → 21.86
```

Pixel loss also decreased overall.

The discriminator loss fluctuated substantially, which is expected in GAN training because the generator and discriminator continuously compete.

---

# 12. Saving the SRGAN

The final models were saved as:

```text
SRGAN_Generator_Final.keras
SRGAN_Discriminator_Final.keras
```

Only the trained generator is required to create the SRGAN-enhanced dataset for Model B.

---

# 13. SRGAN Visual Evaluation

The notebook compares:

```text
LR 32×32
Bicubic 128×128
SRGAN 128×128
Original 128×128
```

Bicubic interpolation provides a conventional resizing baseline.

The SRGAN output is generally sharper and structurally closer to the original HR image than bicubic interpolation.

Important note: SRGAN infers plausible high-resolution details. It cannot perfectly restore information that was completely lost during downsampling.

---

# 14. Creating the Dataset for Model B

After training, the generator was applied separately to the original training and test sets.

Generated images were stored as:

```text
srgan_generated/
    train/
        cat/
        dog/

    test/
        cat/
        dog/
```

The original labels were preserved.

The generated dataset contained:

```text
Training images: 17,451
Testing images:   7,479
```

No test image was used for classifier training.

---

# 15. Model B Data Augmentation

Model B uses the generated SRGAN images.

Training augmentation:

```text
Rotation: ±20°
Width shift: ±10%
Height shift: ±10%
Shear: 10%
Zoom: 10%
Horizontal flipping
```

MobileNetV2-specific preprocessing was applied using:

```python
preprocess_input
```

No random augmentation was applied to the test set.

---

# 16. Model B Architecture

Model B uses the same architecture as Model A:

```text
MobileNetV2 pretrained on ImageNet
      ↓
GlobalAveragePooling2D
      ↓
Dropout(0.3)
      ↓
Dense(128, ReLU)
      ↓
Dense(1, Sigmoid)
```

This ensures that the primary experimental difference between Model A and Model B is the image source.

---

# 17. Model B Transfer Learning

The MobileNetV2 base was initially frozen.

Training settings:

```text
Epochs: 10
Optimizer: Adam
Learning rate: 0.001
Loss: Binary cross-entropy
```

Training accuracy increased to approximately:

```text
96.23%
```

after the initial transfer-learning stage.

---

# 18. Model B Fine-Tuning

The MobileNetV2 base was then made trainable.

All but approximately the last 30 layers were frozen, and BatchNormalization layers were kept frozen.

The effective number of trainable MobileNetV2 layers was 19.

Fine-tuning settings:

```text
Additional epochs: 10
Learning rate: 1e-5
```

Training accuracy increased to approximately:

```text
97.88%
```

---

# 19. Evaluation Metrics

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Binary cross-entropy loss
- Confusion matrix

For final prediction:

```python
y_pred = (y_prob >= 0.5).astype(int)
```

The threshold of `0.5` converts predicted probabilities into binary class predictions.

---

# 20. Final Model B Results

Model B achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.9611 |
| Precision | 0.9513 |
| Recall | 0.9719 |
| F1 Score | 0.9615 |
| ROC-AUC | 0.9945 |
| Loss | 0.1098 |

Confusion matrix:

```text
                 Predicted
                 Cat    Dog

True Cat        3556    186
True Dog         105   3632
```

Correct predictions:

```text
3556 + 3632 = 7188
```

out of:

```text
7479
```

test images.

---

# 21. Final Model A vs Model B Comparison

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | **0.9807** | 0.9611 |
| Precision | **0.9825** | 0.9513 |
| Recall | **0.9789** | 0.9719 |
| F1 Score | **0.9807** | 0.9615 |
| ROC-AUC | **0.9984** | 0.9945 |
| Loss | **0.0616** | 0.1098 |

Model A performed better overall.

However, Model B achieved strong classification performance despite starting from only **32×32 images**.

A 32×32 image contains:

```text
32 × 32 = 1,024 spatial pixels
```

while a 128×128 image contains:

```text
128 × 128 = 16,384 spatial pixels
```

Therefore, the LR image contains only:

```text
1,024 / 16,384 = 6.25%
```

as many spatial pixels as the 128×128 image.

Despite this information reduction, Model B achieved approximately:

```text
96.11% accuracy
```

and:

```text
0.9945 ROC-AUC
```

---

# 22. Interpretation

The experiment does **not** show that SRGAN improves classification compared with directly using the original images.

Instead, it demonstrates that SRGAN can reconstruct useful visual information from strongly downsampled inputs well enough to support high binary-classification performance.

The most important result is:

> Even after reducing images to 32×32, SRGAN-generated 128×128 reconstructions supported a MobileNetV2 classifier with 96.11% test accuracy and 0.9945 ROC-AUC.

---

# 23. Reproduction Steps

To reproduce the experiment:

1. Clone this repository.
2. Create and activate the TensorFlow environment.
3. Install the required Python packages.
4. Place the Cats vs Dogs dataset in the expected dataset directory.
5. Update `BASE_PATH` in the notebook if necessary.
6. Run the notebook from the beginning.
7. Pool all usable images and create the fixed 70/30 stratified split.
8. Train Model A on original images resized to 128×128.
9. Fine-tune Model A.
10. Create 32×32 LR / 128×128 HR image pairs from the training split.
11. Pretrain the SRGAN generator for 5 epochs.
12. Train SRGAN for 150 epochs.
13. Save checkpoints every 5 epochs.
14. Save the final generator and discriminator.
15. Use the trained generator to create SRGAN-enhanced training and test datasets.
16. Train Model B using the same MobileNetV2 classifier architecture.
17. Fine-tune Model B.
18. Evaluate both models on their corresponding test inputs.
19. Compare Accuracy, Precision, Recall, F1, ROC-AUC, loss, and confusion matrices.

---

# 24. Important Files

Typical project structure:

```text
Applied_AI_Midterm/
│
├── Applied_AI_Midterm_SRGAN.ipynb
├── README.md
├── Model_A_vs_Model_B.csv
├── model_A_results.csv
│
├── models/
│
├── checkpoints/
│
├── logs/
│
├── srgan/
│   ├── models/
│   ├── checkpoints/
│   └── logs/
│
└── srgan_generated/
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/
```

Large model files, generated image datasets, logs, and checkpoints may be excluded from GitHub using `.gitignore`.

---

# 25. Key Variables

| Variable | Description |
|---|---|
| `SEED` | Fixed random seed for reproducibility |
| `train_df` | Original 70% training split |
| `test_df` | Original 30% test split |
| `SR_LR_SIZE` | SRGAN low-resolution input size `(32, 32)` |
| `SR_HR_SIZE` | SRGAN target/output size `(128, 128)` |
| `SR_BATCH_SIZE` | SRGAN training batch size |
| `SRGAN_EPOCHS` | Number of SRGAN adversarial training epochs |
| `GENERATOR_PRETRAIN_EPOCHS` | Generator-only pretraining epochs |
| `generator` | Network producing 128×128 images from 32×32 inputs |
| `discriminator` | Network distinguishing real HR from generated SR images |
| `vgg_features` | Frozen VGG19 feature extractor for perceptual loss |
| `p_loss` | Pixel reconstruction loss |
| `perc_loss` | VGG19 perceptual loss |
| `adv_loss` | Generator adversarial loss |
| `gen_loss` | Combined generator loss |
| `disc_loss` | Discriminator loss |
| `sr_train_df` | DataFrame containing generated Model B training images |
| `sr_test_df` | DataFrame containing generated Model B test images |
| `base_model_B` | MobileNetV2 backbone for Model B |
| `model_B` | Complete SRGAN-image classifier |
| `y_prob_B` | Model B predicted probabilities |
| `y_pred_B` | Model B binary predictions |
| `y_true_B` | Ground-truth test labels |
| `model_B_f1` | Model B F1 score |
| `model_B_auc` | Model B ROC-AUC |

---

# 26. Notes on Reproducibility

Exact values may vary slightly across:

- TensorFlow versions
- CUDA/cuDNN versions
- GPU hardware
- random augmentation order
- floating-point implementation differences

Using the same split, random seed, architecture, optimizer settings, preprocessing, and training schedule should produce similar results.

---

## Final Conclusion

The baseline MobileNetV2 classifier operating on directly resized original images achieved the best performance.

The SRGAN-based pipeline nevertheless retained strong discriminative information after severe image downsampling.

This demonstrates that learned super-resolution can produce reconstructed images that remain highly useful for downstream binary classification even when the original input resolution is substantially reduced.

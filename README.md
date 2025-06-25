## 🔍 Project Workflow

### 1. **Data Loading and Preprocessing**

- Images were loaded using `PIL.Image`, converted to RGB.
- All images resized to **128×128** after checking min-max dimensions.
- Labels encoded using `LabelEncoder` and converted to one-hot vectors.
- Dataset was split into training and validation sets (80:20), stratified by class.

### 2. **Model Architecture**

A custom deep CNN was built with the following highlights:

- 4 Convolutional Blocks: `Conv2D → BatchNorm → MaxPooling`
- `GlobalAveragePooling2D` to reduce overfitting
- Dense layers: `128 → Dropout → Softmax(7)`
- Regularization: `Dropout`, `BatchNormalization`
- Loss Function: `categorical_crossentropy`
- Optimizer: `Adam` with `ReduceLROnPlateau` and `EarlyStopping`

### 3. **Data Augmentation**

Augmentation techniques were applied using `ImageDataGenerator`:

- Rotation, shifting, shearing, zoom, horizontal flip
- Helped improve generalization and accuracy

### 4. **Evaluation**

- Accuracy & loss plots for training and validation
- Confusion matrix for class-wise error analysis
- Final validation accuracy: **~70%**

---

## 📊 Results

| Metric           | Value     |
|------------------|-----------|
| Validation Loss  | ~0.90     |
| Validation Accuracy | ~70%   |
| Final Model Size | ~1.6 MB   |
| Best Epoch       | 4         |

---

## 🧠 Key Learnings

- Batch Normalization and GlobalAveragePooling improved convergence and generalization.
- Augmentation stabilized validation accuracy and reduced overfitting.
- Categorical cross-entropy and softmax were correctly used for multi-class classification.
- Proper label encoding and resizing were crucial for training pipeline success.

---

## 📂 Folder Contents

- `waste_classification.ipynb`: Main notebook with all steps
- `data/`: Dataset folder (not uploaded on GitHub)
- `model/`: (Optional) Saved model weights
- `README.md`: Project summary

---

## 🚀 Future Enhancements

- Use class weighting or focal loss to handle any class imbalance
- Experiment with pretrained models (e.g., MobileNet, ResNet) for higher accuracy
- Deploy as a web app or integrate with an IoT-based trash segregation device

---

## 🛠️ Libraries Used

- Python 3.11
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- Scikit-learn
- PIL

---

## 👤 Author

Raj Srivastava  
*This project is part of a CNN assignment focused on sustainable AI solutions.*


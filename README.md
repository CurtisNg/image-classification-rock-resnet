# Image Classification - ResNet Model
This repository documents a **personal learning project** focused on building an image classification model using ResNet-50 to classify rock types. The dataset, sourced from Kaggle, is processed through data cleaning, augmentation, and modeling stages to achieve robust performance. The project includes the data source of the original dataset, and four Jupyter notebooks detailing the workflow: data cleaning, augmentation, modeling, and model application.


---

## Installation Instructions
To run this project locally, follow these steps:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/image-classification-rock-resnet.git
   ```
2. **Set Up a Virtual Environment** (optional but recommended).

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries include `torch`, `torchvision`, `numpy`, `matplotlib`, `pillow`, `albumentations`, `imagehash`, `scikit-learn`, `tqdm`, and `opencv-python`.
5. **Verify Setup**:
   Ensure PyTorch is installed with GPU support (if applicable) by running:
   ```python
   import torch
   print(torch.cuda.is_available())  # Or torch.backends.mps.is_available() for M1 Macs
   ```

---

## Usage Instructions
1. **Explore Notebooks**:
   - `01_data_cleaning.ipynb`: Check data type and removes duplicate images from the dataset.
   - `02_data_augmentation.ipynb`: Balances the dataset with augmentation techniques.
   - `03_image_classification.ipynb`: Trains and fine-tunes the ResNet-50 model.
   - `04_model_application.ipynb`: Applies the trained model to classify random images.
2. **Run Notebooks**:
   Open in Jupyter Notebook.
   Execute cells sequentially, ensuring dataset directories (`Dataset/`) is in the root folder.
3. **Classify New Images**:
   Use `04_model_application.ipynb` by modifying the `random_image_path` or providing a custom image path to the `classify_image` function.

---

## Dataset
- **Source**: [Rock Classification Dataset](https://www.kaggle.com/datasets/salmaneunus/rock-classification?resource=download) from Kaggle.
- **Description**: Contains images of rocks categorized into three main types (Igneous, Metamorphic, Sedimentary) with subcategories (e.g., Basalt, Marble, Sandstone). Total subcategories: 7.
- **Download**: Visit the Kaggle link, download, and extract to the `Dataset/` folder.
- **Processing**:
  - **Original**: Raw dataset with potential duplicates.
  - **Cleaned**: Duplicates removed using average hashing (aHash) in `01_data_cleaning.ipynb`.
  - **Augmented**: Balanced dataset with transformations in `02_data_augmentation.ipynb` to match the largest subcategory count.

---

## Model Training Details
- **Model**: ResNet-50 pre-trained on ImageNet, fine-tuned for 7 rock subcategories.
- **Workflow**:
  1. **Baseline**: Froze all layers except the fully connected (`fc`) layer, trained with the given hyperparameters in `03_image_classification.ipynb`.
  2. **Fine-Tuning**: Unfroze `layer4` and `fc`, used differential learning rates to improve feature adaptation.
- **Data Split**: 80% train, 20% validation & test  (stratified).
- **Training**: Up to 50 epochs with early stopping (patience=10) based on validation loss.
- **Hardware**: Optimized for CPU or GPU.

---

## Conclusion
- **Findings**: 
  - The baseline model showed strengths in classifying distinct rock types but struggled with visually similar subcategories like Marble. Fine-tuning deeper layers (e.g., `layer4`) improved performance by adapting high-level features, particularly for challenging classes. However, a gap between training and validation loss indicates overfitting, and validation loss plateauing suggests limited generalization.
- **Future Work**: 
  - Address overfitting with regularization (e.g., weight decay, dropout). 
  - Enhance generalization by unfreezing additional layers (e.g., `layer3`) with smaller learning rates or applying more diverse data augmentation (e.g., RandomRotation). 
  - Improve specific class performance (e.g., Marble) by collecting more samples or using class-weighted loss. 
  - Conduct hyperparameter tuning to optimize learning rates and scheduler settings.

---

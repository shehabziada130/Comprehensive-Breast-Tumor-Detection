# Comprehensive Breast Tumor Detection 🎗️

Breast cancer is one of the most common cancers worldwide. Early detection and accurate classification of tumors are crucial for improving patient outcomes.  
This project implements an **end-to-end deep learning pipeline** for breast cancer detection and analysis, combining **mammography**, **histopathology**, and **MRI visualization** in a real-world clinical decision sequence.

---

## 📂 Project Structure
- **BCAN_Data_Prepration.ipynb**  
  - Prepares the **CBIS-DDSM mammogram dataset**.  
  - Cleans and merges metadata tables (calcification and mass).  
  - Maps labels (`BENIGN` = 0, `MALIGNANT` = 1).  
  - Matches metadata with corresponding image file paths.  
  - Encodes categorical variables (density, image view, etc.).  

- **BCAN_Models.ipynb**  
  - **Binary classification model** (DenseNet201): benign vs. malignant.  
  - **Multi-label model** predicting multiple pathology attributes (view, side, calc type, distribution, density, assessment).  
  - Image augmentation with OpenCV, PIL, and Keras.  
  - Evaluation using confusion matrices, ROC curves, and metrics.  

- **BCAN_Pathology.ipynb**  
  - Experiment on **IDC Breast Histopathology dataset**.  
  - Binary classification pipeline with ImageDataGenerator.  
  - Provides a second diagnostic checkpoint after mammography.  

- **Streamlit UI**  
  - Integrates all three models in a diagnostic workflow:  
    1. **Mammogram model (CBIS-DDSM)** → Screens mammograms.  
       - If **no suspicious finding** → process ends.  
       - If **suspicious** → forward to pathology stage.  
    2. **Histopathology model (IDC)** → Examines biopsy/pathology images.  
       - If **benign** → request **MRI scan** to visualize tumor region.  
       - If **malignant** → diagnostic process ends with malignant report.  
    3. **MRI visualization**  
       - Tumor area localized and overlaid (heatmap).  
       - Generates a **PDF report** with diagnostic findings and visual evidence.  

---

## 🔬 Datasets
1. **CBIS-DDSM** (Curated Breast Imaging Subset of the Digital Database for Screening Mammography).  
2. **IDC Breast Histopathology Dataset** (Kaggle, Paul Mooney).  
3. **MRI visualization dataset** (for overlays/heatmaps).  

---

## ⚙️ Methods & Techniques
- **Transfer Learning**: DenseNet201 backbone for mammograms & pathology images.  
- **Multi-task learning**: simultaneous label prediction (multi-label model).  
- **Data augmentation**: rotation, zoom, flipping, contrast/brightness shifts.  
- **Visualization**: Grad-CAM / heatmaps for tumor localization on MRI.  
- **Reporting**: Automatic PDF export of findings via Streamlit interface.  

---

## 📊 Results
- Binary classifier successfully separates benign vs. malignant mammograms.  
- Multi-label classifier predicts detailed attributes (view, density, etc.).  
- Pathology classifier provides secondary confirmation.  
- MRI visualization enhances interpretability by overlaying tumor regions.  

📈 **ROC Curves and Metrics**  

🖼️ **Sample Mammogram Overlay**  

🔥 **MRI Heatmap Visualization**  

---

## 🚀 Future Work
- Expand MRI dataset and improve tumor segmentation.  
- Add explainability with SHAP/LIME on pathology predictions.  
- Benchmark against other backbones (EfficientNet, Swin Transformer).  

---

## 🤝 Acknowledgments
- **CBIS-DDSM** dataset from The Cancer Imaging Archive (TCIA).  
- **IDC Breast Histopathology Dataset** by Paul Mooney (Kaggle).  
- **MRI scans** from publicly available datasets.  
- TensorFlow, Keras, OpenCV, Streamlit, scikit-learn open-source communities.  

---

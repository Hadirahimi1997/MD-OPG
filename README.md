# Detection of Pediatric Dental Caries in Panoramic Radiographs Using Artificial Intelligence: A Benchmark Study on MD-OPG

**Authors**: 

Rahimi, Hadi
 
Seyed Mohammadrasoul Naeimi
 
DARVISH, SHAYAN

 
NazemiSalman, Bahareh

 
Razzaghi, Parvin

 
Luchian, Ionut

Budala, Dana-Gabriela



---

## 📌 Overview

In this study, we introduce a Panoramic Radiographs dataset **(MD-OPG)** for Pediatric Dental Caries detection and provide benchmarking experiments for both **classification** and **segmentation** tasks.

We evaluate the performance of standard deep learning models using various loss functions, providing a solid baseline for future research.

---

## 📁 Repository Structure


classification/      # Code for classification task using pretrained ResNet-18 (Keras) + patch extraction code for training this model
segmentation/        # Code for segmentation using UNet and Attention-UNet (Torch) + extracting smile zone images code for these models
figures/             # Supplementary plots (loss and Dice score plots per 3 loss experiments for both segmentation models)

---

## 📊 Supplementary Figures
Training curves (loss and Dice score) for all six segmentation experiments are included in the figures/ directory:

**UNet:**

Focal Loss

Dice Loss

Dice + Focal Loss

**Attention UNet:**

Focal Loss

Dice Loss

Dice + Focal Loss

---

## 📥 Dataset Access

https://doi.org/10.5281/zenodo.17786082

----

## 🤝 Contributions

All authors have contributed to data curation, model development, experimentation, and analysis.

For questions, feel free to open an issue or contact us using the provided email.

---

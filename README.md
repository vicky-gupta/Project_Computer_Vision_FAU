# 🧠 Computer Vision Projects by PR Lab – Summer Term 2024

### 📍 Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

Welcome to my collection of **Computer Vision projects** developed during the **Summer Term 2024** at **PR Lab, FAU Erlangen-Nürnberg**.  
Each project explores a different area of image processing, computer vision, and machine learning — from 3D reconstruction to facial recognition systems.

---

## 🗂️ Project Overview

| # | Project Title | Supervisor | Duration |
|:-:|----------------|-------------|-----------|
| 1 | [📦 Box Detection using RANSAC Algorithm](#-project-1-box-detection-using-ransac-algorithm) | [Dr. Florian Kordon](https://lme.tf.fau.de/person/kordon/) | Apr 2024 |
| 2 | [🎨 Demosaicing of Bayer Patterns and HDR Computation](#-project-2-demosaicing-of-bayer-patterns-and-hdr-computation) | [Dr. Mathias Seuret](https://lme.tf.fau.de/person/seuret/) | May 2024 |
| 3 | [✍️ Writer Identification using SIFT Features](#-project-3-writer-identification-using-sift-features) | [Dr. Vincent Christlein](https://lme.tf.fau.de/person/christlein/) | Jun 2024 |
| 4 | [👤 Face Recognition System using ML Techniques](#-project-4-face-recognition-system-using-ml-techniques) | [Dr. Thomas Koehler](https://ieeexplore.ieee.org/author/37268046800) | Jul 2024 |
| 5 | [🔍 Advanced Object Detection & Face Recognition](#-project-5-advanced-object-detection--face-recognition) | [Mathias Zinnen](https://lme.tf.fau.de/person/zinnen/) | Aug 2024 |

---

## 📦 Project 1: Box Detection using RANSAC Algorithm

<details>
<summary>🔍 Click to expand details</summary>

**Duration:** Apr 2024  
**Associated with:** FAU Erlangen-Nürnberg  

Developed an algorithm to detect and measure **box dimensions using Kinect depth data**.  
The pipeline includes:
- Data loading and preprocessing  
- **RANSAC-based plane detection**  
- Mask filtering and **corner detection**  
- 3D dimension estimation  

**Technologies:**  
`Python` · `NumPy` · `SciPy` · `Matplotlib` · `OpenCV` · `Scikit-Learn`

**Future Enhancements:**
- Handle multiple boxes simultaneously  
- Optimize runtime and memory usage  
- Extend to additional sensors  

**Supervisor:** [Dr. Florian Kordon](https://lme.tf.fau.de/person/kordon/)

</details>

---

## 🎨 Project 2: Demosaicing of Bayer Patterns and HDR Computation

<details>
<summary>📷 Click to expand details</summary>

**Duration:** May 2024  
**Associated with:** FAU Erlangen-Nürnberg  

Implemented a complete pipeline for **demosaicing Bayer patterns** and **HDR image processing**, inspired by Mathias Seuret’s exercises.

Key components:
- Bayer pattern interpretation  
- Demosaicing algorithms and color reconstruction  
- **HDR computation using iCAM06**  
- White balancing and luminosity enhancement  

**Technologies:**  
`Python` · `NumPy` · `RawPy` · `OpenCV`  

**Supervisor:** [Dr. Mathias Seuret](https://lme.tf.fau.de/person/seuret/)

</details>

---

## ✍️ Project 3: Writer Identification using SIFT Features

<details>
<summary>🖋️ Click to expand details</summary>

**Duration:** Jun 2024  
**Associated with:** FAU Erlangen-Nürnberg  

Developed a **writer identification system** using the ICDAR17 dataset.  
Integrated **Bag of Visual Words**, **VLAD encoding**, **PCA whitening**, and **SVM classification** for robust performance.

**Key Modules:**
- SIFT feature extraction (`OpenCV`)  
- MiniBatchKMeans for codebook generation  
- VLAD encoding + power normalization  
- LinearSVC for classification  

**Technologies:**  
`Python` · `NumPy` · `Scikit-Learn` · `OpenCV` · `VLAD` · `PCA`

**Supervisor:** [Dr. Vincent Christlein](https://lme.tf.fau.de/person/christlein/)

</details>

---

## 👤 Project 4: Face Recognition System using ML Techniques

<details>
<summary>🧩 Click to expand details</summary>

**Duration:** Jul 2024  
**Associated with:** FAU Erlangen-Nürnberg  

Developed a **face recognition and re-identification system** using both **supervised and unsupervised learning**.  

Key modules:
- Face detection and alignment via **MTCNN**  
- Feature extraction using **FaceNet**  
- Classification using **k-NN (closed-set & open-set)**  
- Clustering with **k-Means** for unsupervised recognition  
- Evaluation via **DIR curves**  

**Advanced Additions:**
- Single and Multi Pseudo Label (SPL/MPL) methods for open-set recognition  

**Technologies:**  
`Python` · `NumPy` · `SciPy` · `Scikit-Learn` · `MTCNN`

**Supervisor:** [Dr. Thomas Koehler](https://ieeexplore.ieee.org/author/37268046800)

</details>

---

## 🔍 Project 5: Advanced Object Detection & Face Recognition

<details>
<summary>🚀 Click to expand details</summary>

**Duration:** Aug 2024  
**Associated with:** FAU Erlangen-Nürnberg  

Built a **hybrid system** combining **object detection (Selective Search)** and **face recognition (MTCNN + FaceNet)**.  
Implemented **open-set recognition** with pseudo-labeling to improve generalization and robustness.

**Core Features:**
- Region proposal via **Selective Search**  
- **MTCNN-based** face detection and alignment  
- **FaceNet embeddings** + `k-NN` for classification  
- **DIR curve evaluation** for performance metrics  

**Technologies:**  
`Python` · `NumPy` · `Scikit-Learn` · `OpenCV` · `Deep Learning`  

**Supervisor:** [Mathias Zinnen](https://lme.tf.fau.de/person/zinnen/)

</details>

---

## 🧰 Tech Stack Overview

| Category | Tools & Libraries |
|-----------|-------------------|
| 🧠 Programming | Python |
| 🧮 Computation | NumPy · SciPy |
| 📊 Visualization | Matplotlib |
| 🧑‍💻 ML / CV | Scikit-Learn · OpenCV · VLAD · PCA |
| 🤖 Deep Learning | MTCNN · FaceNet |
| 📷 Image Processing | RawPy · HDR (iCAM06) |

---

## 📫 Contact

Feel free to connect or reach out for collaborations or research discussions!

**Author:** *Vicky Vicky*  
**Institution:** *Pattern Recognition Lab, FAU Erlangen-Nürnberg*  
**Email:** *vicky.vicky@fau.de*  
**GitHub:** [github.com/vicky-gupta](https://github.com/vicky-gupta)

---

⭐ *If you find these projects useful, please consider giving this repo a star!*

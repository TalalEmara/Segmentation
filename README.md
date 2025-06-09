<h1 align="center">
    <img alt="Image Segmentation Demo" src="Readme/demo.gif" />
</h1>

<h1 align="center">Image Segmentation</h1>
<div align="center">
  <img src="https://github.com/user-attachments/assets/e3870856-c43d-435f-84f6-480a1a699796" >
</div>

<h4 align="center"> 
	Status: ✅ Completed
</h4>

<p align="center">
 <a href="#about">About</a> •
 <a href="#features">Features</a> •
 <a href="#tech-stack">Tech Stack</a> •  
 <a href="#developers">Developers</a>
</p>

---

## 🧠 About

The **Image Segmentation** project implements various advanced image segmentation techniques including thresholding methods, clustering algorithms, and region-based approaches. It provides a comprehensive toolkit for separating images into meaningful regions based on pixel intensities, colors, and spatial relationships.

This tool serves as a foundational platform for tasks in:
- Medical image analysis
- Object recognition
- Computer vision pipelines
- Image editing and processing

---

## ✨ Features

### 🔳 Thresholding Methods
- **Otsu Thresholding**
  - Automatic optimal threshold calculation
  - Maximizes between-class variance
  - Effective for bimodal histograms
 
 <div align="center">
  <img src="https://github.com/user-attachments/assets/66c518cb-a9ce-4739-9387-714fb8a9add7">
 </div>
  
- **Optimal (Iterative) Thresholding**
  - Adaptive threshold calculation
  - Convergence-based stopping criterion
  - Handles varying illumination
 <div align="center">
  <img src="https://github.com/user-attachments/assets/18b61414-efc3-4513-8b84-2e80adefa589">
 </div>
- **Spectral (Multi-Otsu) Thresholding**
  - Multi-class segmentation
  - Extends Otsu's method for N classes
  - Captures finer intensity variations
 <div align="center">
  <img src="https://github.com/user-attachments/assets/1fb41dbb-f97c-4c73-a02f-d19ff44af972">
 </div>

- **Local Thresholding**
- Patch-based adaptive thresholding
- Supports all global methods locally
- Handles non-uniform illumination
- Adjustable patch sizes


### 🟣 Clustering Techniques
- **K-means Clustering**
  - Color-based segmentation
  - Adjustable cluster count (K)
  - Euclidean distance metric

- **Mean Shift Clustering**
  - Non-parametric clustering
  - Automatic cluster discovery
  - Bandwidth parameter control

**Clustering Results:**
<div align="center">
  <img src="https://github.com/user-attachments/assets/34a7d5a1-2149-45c7-a490-314ce1a6198b" width="45%">
  <img src="https://github.com/user-attachments/assets/798ea1bd-a4e6-4bfd-8e05-3fad9dea6c6a" width="45%">
</div>

- **Region Growing**
  - Seed-based segmentation
  - Intensity/color similarity thresholds
  - 8-directional neighborhood expansion
<div align="center">
  <img src="https://github.com/user-attachments/assets/2d0c13c0-93ea-49d4-ad66-2b3f4bd94d33" width="45%">
  <img src="https://github.com/user-attachments/assets/5cff9697-baf3-4faa-b78b-6e4cfedae0ce" width="45%">
</div>


- **Agglomerative Clustering**
  - Hierarchical superpixel merging
  - Combines color and spatial features
  - SLIC superpixel initialization




---

## ⚙️ Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **SciPy**
- **scikit-image**
- **Matplotlib**

---

## 👨‍💻 Developers


## Developers

| [**Talal Emara**](https://github.com/TalalEmara) | [**Meram Mahmoud**](https://github.com/Meram-Mahmoud) | [**Maya Mohammed**](https://github.com/Mayamohamed207) | [**Nouran Hani**](https://github.com/Nouran-Hani) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|
---

## 📎 Learn More

* [Otsu's Method](https://en.wikipedia.org/wiki/Otsu%27s_method)
* [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
* [Region Growing Segmentation](https://www.sciencedirect.com/topics/engineering/region-growing)
* [Mean Shift Clustering](https://en.wikipedia.org/wiki/Mean_shift)

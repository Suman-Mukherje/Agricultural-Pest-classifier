# Agricultural-Pest-classifier

## Overview
Web application for Aggricultural pest classification using deep learning model
This application can only classify 12 types of agricultural pest which are "ants", "bees", "beetle", "catterpillar", "earthworms", "earwig", "grasshopper", "moth", "slug", "snail", "wasp", "weevil".

The model used for classification is **MobileNetV2**. The dataset was sourced from Kaggle: [Dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset).

Key libraries used in this project include **PyTorch**, **PIL (Pillow)**, and **Flask**.

## Getting Started

To get a local copy of this repository up and running, follow these steps:

### 1. Clone the Repository

```bash
git clone ## Getting Started

To get a local copy of this repository up and running, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Suman-Mukherje/Agricultural-Pest-classifier.git
```

### 2. Navigate to the Project Directory

```bash
cd pestclassifier
```
### 3. Install all required packages

```bash 
pip install Flask torch torchvision Pillow
```
### 4. Start the App

```bash
python app.py
```

### 5. View the web application

After starting app in terminal there is a http server address like 'http://127.0.0.1:5000/' copy the link and paste in browser , a web application will be shown.


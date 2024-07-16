# Real-Time-Object-Detection-for-PPE-Personal-Protective-Equipements-with-Yolov5-Yolov9

## Table of Contents

01. [Project Overview](#1-project-overview)
02. [Key Objectives](#2-key-bjectives)
03. [Technologies Used](#3-technologies-used)
04. [Methodology](#4-methodology)
05. [Key Components](#5-key-components)

07. [How to Run](#7-how-to-run)
08. [Acknowledgment](#8-acknowledgment)
09. [Conclusion](#9-conclusion)
10. [Contact](#10-contact)

## 1. Project Overview
This project focuses on utilizing YOLOv5/YOLOv9 for real-time detection of Personal Protective Equipment (PPE) in various settings. The detection of PPE items such as helmets, vests, masks, and gloves is crucial for ensuring safety compliance in industrial and healthcare environments.

## 2. Key Objectives

###### Accurate Detection: 
Develop a model capable of accurately detecting various types of PPE items in diverse environments.
###### Real-Time Performance: 
Ensure the system operates in real-time, meeting the demands of dynamic environments.
###### Model Optimization: 
Fine-tune YOLOv5/YOLOv9 models to achieve optimal detection performance for PPE items.

## 3. Technologies Used

- Python
- PyTorch
- YOLOv5/YOLOv9
- Roboflow (for dataset management)
- CUDA (for GPU acceleration)

## 4. Methodology

The project adopts a transfer learning approach using YOLOv5/YOLOv9 pre-trained models. The methodology includes data collection, model configuration using YAML files, and training with GPU acceleration to achieve real-time PPE detection capabilities.


## 5. Key Components

###### Data Collection
Data sourced from Roboflow, consisting of annotated images depicting PPE items like helmets, vests, masks, and gloves.

###### Transfer Learning
Utilization of pre-trained YOLOv5/YOLOv9 models for transfer learning on the PPE dataset to expedite training and enhance detection accuracy.

###### Model Training
Training conducted on GPUs for accelerated computation, with parameters such as batch size, learning rate, and epochs optimized for performance.

## 7. How to run
1. **Clone the Repository:**

   Use the following command to clone the repository:

   ```bash
   git clone git@github.com:AMEZIANDouha/Real-Time-Object-Detection-for-PPE-Personal-Protective-Equipements-with-Yolov5-Yolov9.git
2. **Install Dependencies:**
    Navigate to the project directory and install dependencies using:
     ```bash
     pip install -r requirements.txt
4. **Training:**
   To train the model, run:
       ```bash
   python train.py --data data.yaml --cfg yolov5.yaml --weights yolov5s.pt --batch-size 16 --epochs 50

6. **Validation:**
    Evaluate the model using the validation set:
       ```bash
   python val.py --data data.yaml --weights runs/train/exp/weights/best.pt

8. **Testing:**
     Test the model's performance on new data:

    ```bash
    python test.py --data data.yaml --weights runs/train/exp/weights/best.pt --img-size 640 --conf-thres

## 8. Acknowledgment

We acknowledge the contributions of the Roboflow team for providing the annotated PPE dataset and the open-source community for developing PyTorch and YOLOv5/YOLOv9.

## 9. Conclusion
In conclusion, the "Real-Time Object Detection for PPE with YOLOv5/YOLOv9" project demonstrates significant advancements in leveraging deep learning for enhancing workplace safety and compliance through efficient PPE detection. By utilizing state-of-the-art models and GPU acceleration, the system offers accurate and real-time detection capabilities.

## 10. Contact

For any inquiries or further information regarding the "Real-Time Object Detection for PPE with YOLOv5/YOLOv9" project, please feel free to reach out:

- **Name      :**    AMEZIAN Douha  
- **Email     :**   [ameziandouha9@gmail.com](ameziandouha9@gmail.com)  
- **LinkedIn  :**  [AMEZIAN Douha](https://www.linkedin.com/in/douha-amezian-033629280/)  

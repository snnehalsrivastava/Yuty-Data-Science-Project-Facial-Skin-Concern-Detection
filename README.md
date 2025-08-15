# Multi Facial Skin Concern Detection Model Using Advanced ResNet50 U-Net Model

# Project carried out for YUTY.LTD 

## Project Overview

This project delivers an **innovative AI-driven solution for the precise detection and segmentation of nine distinct facial skin imperfections directly from user selfies**. Developed in collaboration with **Yuty**, a visionary beauty enterprise, this system addresses a critical need for **personalization in the rapidly expanding beauty industry**, which is projected to reach over $101 billion in revenue by 2027.

Yuty's mission is to alleviate the "paradox of choice" that consumers face amidst an overwhelming array of skincare products by offering **highly tailored and personalized beauty experiences** through advanced machine learning. Our solution empowers Yuty to deliver a suite of enhanced services, including personalized skincare advice, virtual product try-ons, remote dermatology consultations, and comprehensive skincare progress tracking. This strategic initiative is designed to significantly elevate customer experience, satisfaction, and loyalty, positioning Yuty at the **forefront of the competitive beauty industry landscape**.


## Problem Statement

The central business problem addressed by this project is the **establishment of an automated and reliable machine learning model capable of accurately identifying and highlighting nine specific facial skin imperfections from user-uploaded selfies**. These nine concerns are: **Redness, Wrinkles/Fine Lines, Dark Circles, Uneven Skintone, Clogged Pores, Hyperpigmentation/Dark Marks (HDM), Blemish, Acne, and Freckle/Beauty Spot**. The system aims to correct consumers' self-perception of skin issues, leading to more precise and effective product recommendations.

### Key Research Questions Guiding Our Work

1.  **Model Suitability**: Is U-Net and its variants suitable for skin concern detection?
2.  **Architectural Choices**: Identification of other suitable deep learning architectures.
3.  **Data Size Impact**: The influence of dataset size on model performance.
4.  **Imbalanced Datasets**: Solutions for addressing class imbalance among the nine skin concerns and between skin concern areas vs. non-skin concern areas.
5.  **Ensemble Learning**: The impact of ensemble learning on model enhancement.

## Research Objectives

Our primary research objectives were threefold, aiming for a model that is both highly effective and commercially viable:

*   **Accuracy**: To attain **effective accuracy in recognizing all nine specified facial skin concerns**.
*   **Efficiency**: To ensure the model's efficiency for **real-time processing of user selfies**, enabling quick and seamless user experiences.
*   **Generalizability**: To cater to a **wide range of demographics and skin types**, accounting for variations in lighting, skin tones, and facial expressions.

## Key Features & Business Applications

The successful development of this deep learning model enables several pivotal applications for Yuty and the broader beauty industry:

*   **Personalized Skincare Recommendations**: Accurately detecting concerns like acne, dark spots, or redness to provide **tailored product recommendations** that precisely meet individual customer needs.
*   **Virtual Try-On for Skincare Products**: Integrating real-time skin concern highlights with virtual try-on features, allowing customers to **visualize product effectiveness** before purchase.
*   **Remote Dermatology Consultations**: Assisting dermatologists by **pre-analyzing selfies** and flagging potential concerns, thereby streamlining the diagnostic process.
*   **Skincare Progress Tracking**: Enabling users to **monitor the effectiveness of their skincare routines over time** through regular selfie uploads.
*   **Enhanced Customer Satisfaction & Loyalty**: By providing accurate diagnoses and customized recommendations, the system is expected to increase repurchase rates and generate positive word-of-mouth referrals.
*   **Streamlined Customer Support**: Empowering users with self-service tools can **reduce customer support tickets and lower operational costs** for Yuty.
*   **Competitive Advantage & Market Leadership**: Positioning Yuty as an innovative, cutting-edge brand in the personalized beauty space, potentially leading to a **sales uplift of up to 10%**.

## Methodology

Our methodology centered on the **U-Net convolutional neural network**, a architecture renowned for its proficiency in medical image segmentation and efficiency with limited datasets.

### 1. Data Collection & Pre-processing

*   **Dataset Source**: The project utilized a proprietary dataset of **22,132 images** provided by Yuty, accompanied by a JSON file containing annotations for skin concern areas.
*   **Data Cleaning**: A rigorous two-stage cleaning process was implemented:
    *   **Manual Screening**: **14,515 images** remained after initial manual screening, achieving a 65.6% retention rate.
    *   **Data Quality Filtering**: Further filtering based on image quality characteristics (mean pixel value, variance, noise) ensured optimal dataset integrity, resulting in **327 images for initial training** and later **expanded to 1,000 samples** for overall model optimization.
*   **Image Transformation**: Original 1024x1024 pixel images and their corresponding multi-layer masks (one layer for each of the nine skin concerns) were **cropped into smaller 256x256 windows** with a stride of 256. This cropping was performed to optimize U-Net's performance, which is better suited for smaller input sizes. Pixel values were normalized from `` to ``.

### 2. Model Selection & Architecture

*   **Core Architecture - U-Net**: U-Net was selected due to its proven efficacy in biomedical segmentation, its ability to handle intricate structures and variations common in medical images, and its **data and computational efficiency**, especially critical given the constraints of Google Colab. Its adaptable encoder-decoder structure also allows for easy integration with pre-trained models via transfer learning.
*   **Encoder Comparison**: An extensive comparison of eight experimental groups involving U-Net with various pre-trained encoders was conducted, including **VGG16, Xception, ResNet50, ResNet101, EfficientNetB4, and DenseNet201**.
    *   **ResNet50 was identified as the most robust encoder** for subsequent model enhancements, demonstrating superior learning capabilities and a strong balance between complexity and computational efficiency. Its pre-trained weights significantly boost performance, particularly with limited labeled data. EfficientNetB4 also showed promising generalization capabilities.

### 3. Evaluation Metrics

Model performance was rigorously evaluated using key segmentation metrics:

*   **Intersection over Union (IoU)**: A primary metric measuring the overlap between predicted and ground-truth regions, crucial for semantic segmentation.
*   **Dice Coefficient**: A similarity measure often used for evaluating image segmentation, especially valuable for imbalanced classes.
*   **Recall**: Measures the proportion of actual positive instances (skin concerns) correctly identified, which is vital in dermatological applications to avoid missing issues.

### 4. Loss Function

**Focal Loss** was chosen as the primary loss function. This was crucial for addressing the **inherent class imbalances** within the dataset, specifically the disparity between skin concern areas and non-skin concern areas, as well as the imbalance among different skin concern types (e.g., under-represented conditions like Acne and HDM, which initially showed 0% accuracy). Focal loss assigns a higher weight to hard-to-classify samples, ensuring that rare conditions receive adequate attention during training.

### 5. Overfitting Mitigation & Model Enhancement Strategies

Several advanced strategies were employed to enhance model performance and combat overfitting, ultimately leading to a **14% improvement in IoU** compared to the initial model version:

*   **Dataset Expansion**: Increasing the training data from 327 to **1,000 samples** proved effective in enhancing prediction accuracy for more prevalent skin concerns like Redness, Dark Circles, and Wrinkles Fine Lines.
*   **Addressing Data Imbalance**: For under-represented classes such as Acne (5.8% of Yuty's total images) and HDM (7.1%), **ensemble learning** was implemented. A dedicated enhancement model for **Acne**, trained on an additional **579 Acne-specific sample images**, led to a **significant surge in prediction accuracy from 0% to 27%** for Acne, and from 0% to 3% for HDM, using the recall metric. Weight initialization was also explored to balance skin concern areas versus non-skin concern areas in the loss function.
*   **Facial Region Detection**: A pre-trained **MTCNN face detection model** was integrated to filter out erroneous predictions made outside the facial area, thereby improving the practical relevance for end-consumers and yielding an **IoU increment of 0.07%**. This eliminated predictions on hair or background elements.
*   **Hyperparameter Tuning**: Techniques such as **L2 Regularization** (with an optimal value of 0.0001) and **Dropout** were strategically applied to enhance model robustness and curb overfitting.
*   **Decoder Streamlining**: Experiments with simplified decoder designs were conducted, though the original complex decoder was retained for better feature identification across all nine skin concerns, as simpler structures led to decreased performance.

## Key Findings and Results

Our refined **ResNet50-based U-Net model** achieved significant progress, demonstrating a **14% improvement in IoU** compared to its initial version. The model particularly **excels in detecting prominent skin concerns** such as **Redness, Wrinkles Fine Lines, and Dark Circles**.

The optimal model performance, integrating all optimizing strategies, is summarized below:

| Skin Concern Class              | IoU  | Dice Coefficient | Recall |
| :------------------------------ | :--- | :--------------- | :----- |
| Redness                         | 0.45 | 0.62             | 0.65   |
| Wrinkles_Fine_Lines (WFL)       | 0.42 | 0.59             | 0.60   |
| Dark_Circles                    | 0.47 | 0.64             | 0.58   |
| Blemish                         | 0.27 | 0.43             | 0.39   |
| Clogged_Pores                   | 0.23 | 0.37             | 0.37   |
| Acne                            | 0.07 | 0.13             | **0.27** |
| Freckle_Beauty_Spot (FBS)       | 0.14 | 0.25             | 0.20   |
| Uneven_Skintone                 | 0.16 | 0.27             | 0.19   |
| Hyperpigmentation_Dark_Marks (HDM) | 0.03 | 0.05             | 0.03   |
| **Average**                     | **0.25** | **0.37**         | **0.36** |

_Note: IoU, Dice Coefficient, and Recall are presented here as averages over the final test dataset [Table 10]. Recall for Acne improved to 27% due to the ensemble learning strategy._

## Limitations & Future Development

Despite the significant advancements, the project encountered several limitations that highlight avenues for future research and improvement:

*   **Computational Constraints**: The memory and unit limitations of Google Colab significantly restricted the volume of training data and complexity of models that could be processed. Future work would greatly benefit from migrating to **high-performance computing clusters like Summit or Sierra**.
*   **Data Underrepresentation**: A persistent challenge was the insufficient data for specific skin concerns, particularly **Acne and Hyperpigmentation Dark Marks**. These classes constituted only 5.8% and 7.1% respectively of Yuty's total 22,132 images, hindering the model's learning capability despite ensemble efforts.
*   **Noisy-labelled Annotations**: Inaccuracies in the AI-marked annotations occasionally compromised the precision of the model's training.
*   **Generalizability Limitations**: The restricted sample size potentially impacted the model's adaptability to a wider array of lighting conditions, skin tones, and facial expressions.

Future endeavors will focus on:

*   **Extensive Dataset Expansion**: Especially for underrepresented classes, exploring more relevant data samples, data synthesis, or utilizing pre-trained models specific to these concerns.
*   **Improved Annotation Quality**: Investigating the use of dermatologist-annotated datasets or advanced noise reduction techniques for cleaner, more accurate ground truth data.
*   **Advanced Architectures**: Further exploration of alternative cutting-edge architectures like the Multi-scale Dense U-Net or Attention U-Net to further optimize the balance between model complexity and generalization.
*   **Model Pruning**: Implementing pruning techniques to remove extraneous or less relevant data can further enhance model efficiency and accuracy.
*   **Addressing Label Quality**: Deploying networks that critically evaluate label quality, drawing inspiration from existing research.

## Technologies Used

*   **Programming Language**: Python
*   **Deep Learning Frameworks**: TensorFlow, Keras
*   **Core Model Architecture**: U-Net
*   **Encoders**: ResNet50 (primary), VGG16, Xception, ResNet101, EfficientNetB4, DenseNet201
*   **Image Processing Libraries**: NumPy, PIL (Pillow), OpenCV, scikit-image
*   **Machine Learning Utilities**: scikit-learn
*   **Data Visualization**: Matplotlib
*   **Face Detection**: MTCNN

## Authors

*   Snnehal Srivastava
*   Saleha Zahid
*   Senkun Xiang
*   Zhuoxin Ye
*   Xi Ye

**Supervisors:** Chris Sutton, Lei Fang, Natalia Efremova

## Acknowledgments

This project was developed by MSc Business Analytics 2022/2023 Group 9 under the guidance of our esteemed supervisors: **Chris Sutton, Lei Fang, and Natalia Efremova**. We extend our gratitude to Yuty for providing the dataset and the opportunity to work on this impactful project.


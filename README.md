#  Quantum Facial Biometrics

A hybrid **Classical + Quantum Facial Biometrics** system for secure and intelligent face detection and recognition.  
This project combines **classical machine learning (PCA, Haar Cascade, cosine similarity)** with a **Quantum Variational Circuit (VQC)** using **Pennylane** to enhance face detection and recognition accuracy.

---

##  Flowchart Overview
            ┌────────────────────────┐
            │  Dataset Preparation   │
            │ (CelebA + CIFAR-10)    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ Image Enhancement(CLAHE│
            │ + Gamma Correction)    │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ ROI Extraction (Haar)  │
            │ & PCA (8 Components)   │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ Quantum Face Detection │
            │ (VQC - 8 Qubits)       │
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ Recognition Database   │
            │ PCA (512) + Cosine Sim.│
            └───────────┬────────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ Real-Time Recognition  │
            │ (prediction_new.py)    │
            └────────────────────────┘
---
##  Folder Structure
```
Quantum-Facial-Biometrics/
│
├── Dataset_preparation.py         # Step 1 - Download and prepare datasets
├── Face_DB2.py                    # Step 2 - Train quantum detection model
├── db_creation2.py                # Step 3 - Build recognition database
├── prediction_new.py              # Step 4 - Real-time recognition
│
├── .kaggle/
│ ├── kaggle.json                  #json file should contain your API key
├── Dataset25k/
│ ├── faces/
│ ├── nonfaces/
│ ├── enhanced_faces/
│ ├── enhanced_nonfaces/
│ ├── ROI_faces/
│ ├── ROI_nonfaces/
│ └── qc_class_roi/ #You can change this as Office ROI i.e,images in this folder are the roi of office staff
│
├── vqc_face_model_roi.pth                       #Used for Face or Not Face verification
├── pca_detection_roi.pkl                        #Used for Face or Not Face verification
├── qcclass_recognition_db_cosine_roi.pkl        #this is the db_file
├── qcclass_scaler_recognition_cosine_roi.pkl    #it save thes scalar file
├── qcclass_pca_recognition_cosine_roi.pkl       #saves the pca components of the databse
├── qcclass_used_paths_cosine_roi.pkl            #saves the paths of the db.
└── qcclass_roi_paths_cosine_roi.pkl             #will save the ROI_paths of the database
```
---
##  Kaggle API Setup (Required for CelebA Dataset)

The **CelebA dataset** is downloaded automatically from Kaggle using their API.

### 1. Get Your Kaggle API Key
1. Go to your Kaggle account: [https://www.kaggle.com/](https://www.kaggle.com/)
2. Click your **Profile → Account → Create API Token**
3. A file named `kaggle.json` will be downloaded.

### 2. Save the API Key
    Place the file in the following location as given in folder structure.
---
## Environment Setup

### 1. Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```
### 2.Install Dependencies

```
pip install -r requirements.txt
```
----

## Step by step Execution
### Step 1 — Dataset Preparation

  Script: dataset_preparation.py
  
  This step:
    
    Downloads CelebA dataset (faces) #This may take max by upto 1hr to dowload, depends on the internet connectivity
    
    Extracts CIFAR-10 (non-faces)  #This is done directly from python library 
    
    Creates 25k balanced samples for each  #We set the limit for face images as 35k in the code as we loose some images in preprocessing.
    After Preprocess,we are using 25k images for each of them.
    Total 50k images are used for training.

 Run:
   ```
    python dataset_preparation.py
  ```
Output Folders:
  ```
    Dataset25k/faces/
    Dataset25k/nonfaces/

  ```
### Step 2 — Train Quantum Detection Model

  Script: Face_DB2.py
  
  This step:
  
    Enhances dataset images using CLAHE + Gamma Correction
    
    Extracts ROI for faces using Haar Cascade
    
    Reduces dimensions with PCA (8 components)
    
    Trains a Quantum Variational Circuit (VQC) using Pennylane
    
  Run:
    ```
    python FaceDB2.py
    ```
Output Files:
    ```
      pca_detection_roi.pkl
      vqc_face_model_roi.pth
    ```
### Step 3 — Recognition Database Creation

  Script: db_creation2.py
  
  This step:
  
    Now, how we create a database is:
    1. First, capture the minimum 100 unique identities of persons at least 2-3 images per person. 
    2. By default, it will be saved in the local camera folder. Save it in the code repository folder and give it any name (e.g., Quantum-Facial-Biometrics). 
    3. Use the same folder name in the enhance.py code and set an output folder name(as your convenience) so that enhanced images will be saved. 
    4. Once this is done, use the output enhanced folder in the db_creation.py file and run it.

    Now run the file
        Uses your captured & enhanced faces (100+ unique identities)
        Extracts ROI from each
        Builds PCA (512 components) recognition database
        Saves database with prefix-controlled filenames

Run:
```
python db_creation2.py

```
Output Files:
```
qcclass_pca_recognition_cosine_roi.pkl
qcclass_scaler_recognition_cosine_roi.pkl
qcclass_recognition_db_cosine_roi.pkl
qcclass_used_paths_cosine_roi.pkl
qcclass_roi_paths_cosine_roi.pkl
```
### Prefix Usage:
The prefix (e.g., qcclass_, new_) in filenames lets you manage multiple database versions.
Set it in db_creation.py: 
```
OUTPUT_PREFIX = "qcclass_"
#Change it to "new_" or any custom tag to load alternate databases during testing.  
```
### Understanding the Quantum Layer

  The Quantum Variational Circuit (VQC) replaces classical neural network layers with parameterized quantum gates.
  These gates learn complex feature relationships within face embeddings using quantum interference and entanglement.

Architecture:

    8 Qubits
    
    AngleEmbedding for input mapping
    
    StronglyEntanglingLayers for feature learning
    
    TorchLayer for hybrid quantum-classical training
### Outputs and Artifacts
| File / Folder                               | Description                              |
| ------------------------------------------- | ---------------------------------------- |
| `pca_detection_roi.pkl`                     | PCA model for 8-component face detection |
| `vqc_face_model_roi.pth`                    | Trained Quantum VQC model                |
| `qcclass_pca_recognition_cosine_roi.pkl`    | PCA model for 512-component recognition  |
| `qcclass_recognition_db_cosine_roi.pkl`     | Recognition embeddings database          |
| `qcclass_scaler_recognition_cosine_roi.pkl` | Feature scaler for recognition           |
| `qc_class_roi/`                             | Enhanced ROI images used for recognition |

---

##  Summary

| Component | Technique / Algorithm | Output Files |
|------------|----------------------|---------------|
| **Dataset Preparation** | CelebA + CIFAR-10 preprocessing | `Dataset25k/faces/`, `Dataset25k/nonfaces/` |
| **Face Detection** | PCA (8 components) + Quantum Variational Circuit (VQC) | `vqc_face_model_roi.pth`, `pca_detection_roi.pkl` etc..,.|
| **Recognition Database** | PCA (512 components) + Cosine Similarity | `qcclass_pca_recognition_cosine_roi.pkl`, `qcclass_recognition_db_cosine_roi.pkl`etc..,.|
| **Prediction** | Real-Time Webcam Recognition | Shows the results |
| **Quantum Backend** | Pennylane `default.qubit` simulator | Classification for Face or NotFace |

---

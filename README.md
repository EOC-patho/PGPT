# PGPT
## **Overview**
This is the primary repository for reproducing the PGPT framework, a self-supervised deep learning model designed for analyzing histopathological whole slide images (WSIs). PGPT supports tasks such as feature extraction, tile-level and slide-level predictions, and survival outcome analysis.  
The model utilizes a generative self-supervised learning approach, integrating attention-based mechanisms to provide interpretable predictions.
## **Installation**
1. Install dependencies:  
conda install pytorch torchvision torchaudio -c pytorch
pip install numpy、matplotlib、scikit-learn、opencv-python==4.6.0.66、tensorflow、keras、tensorwatch、einops、timm
2. Installation time is approximately 1 hour.
## **Pipeline Overview**
1. Preprocessing
Run the slide_preparation script to prepare WSI datasets:
 Extract tiles from WSIs.
 Filter tiles based on quality and size.
2. Self-Supervised Pretraining
Perform self-supervised pretraining to extract upstream features:
 Train the model using a dataset of WSIs with unlabeled tiles.
 Save the final weights for feature extraction.
3. Tile-Level Supervised Learning
Train on tile-level tasks using labels for specific tasks:
 Use features extracted from self-supervised weights.
4. Feature Extraction
Reconstruct and extract tile-level features:
 Extract features using pre-trained weights from step 2 and tile labels.
5. Slide-Level Supervised Learning
Train on slide-level tasks (e.g., survival prediction, target prediction):
 Use features aggregated from the tile-level tasks.
## **Running process and outputs**
Run time is about 6*A800*168h for self-supervised learning and 6*A800*96h for classification task.

Step 1: Up13 Pretraining

  The Up13 model is pre-trained on a large collection of WSIs using self-supervised learning to capture general histopathological features.

Step 2: Dataset Preparation

  The cohort is split at the patient level into training, internal validation, and external validation sets.

  Three unlabeled .db files are created for these datasets: training set, internal validation set, and external validation set.

Step 3: Feature Extraction and Reconstruction with Up4

  The Up4 model is initialized using the weights from Up13.

  Up4 extracts features and performs reconstruction on the unlabeled training and internal validation .db files.

Step 4: Training with Up57

  The Up57 model is trained using the features extracted by Up4 from the unlabeled .db file of the training set.

  The model starts with random weights for this step.

Step 5: Training, Feature Extraction, and Reconstruction with Low4(tile level supervised learning)

  The Low4 model is initialized again using the weights from Up13.

  Low4 is trained on the labeled .db files (training and internal validation sets).

  After training, Low4 extracts features and performs reconstruction on the same labeled .db files.

Step 6: Training with Low57

  The Low57 model is trained using the features extracted by Low4 from the labeled .db file of the training set.

  The training process uses the best-performing weights obtained from the Up57 model.

Step 7: Final Inference with Low57 on Task 8(WSI level supervised learning)

  Using the best weights from the Low57 model, the system performs the final inference on Task 8.
## **Outputs**
DB Files:

  Feature-extracted .db files for labeled and unlabeled datasets.

Model Weights:

  Best-performing weights for Up57, up4 and Low57, used in downstream analysis.

Slide-Level Predictions:

  Final classification results, including probabilities for favorable or poor outcomes.

Visualization:

  Attention maps and reconstructed features are saved for interpretation and analysis.

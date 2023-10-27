# Siamese Neural Network for Image Similarity

## What is Siamese Neural Network?

A Siamese Neural Network is an artificial neural network that consists of two or more similar subnetworks. These subnetworks have the same parameters with the same weights. The network is designed to compare feature vectors and determine the similarity of inputs. Identical deep convolutional neural networks (CNNs) are trained in a Siamese network design to obtain feature vectors classifying between samples of each image class, which are then compared to validate the similarity of the input images. For example, if you are given two images, the Siamese Neural Network helps find the relation between these two images and whether they are similar.

---

## Objective 

To build Siamese Neural Network for Image Similarity application.

---

## Why is CNN preferred over other networks in Siamese?

The Convolutional Neural Network (CNN or ConvNet) is a subtype of neural network mostly utilized for image and speech recognition applications. The built-in convolutional layers reduce picture dimensionality without compromising information. The pre-processing required in a CNN is much less than in other classification algorithms, making CNN the best choice to build a Siamese network.

---

## Understanding the Triplet Loss Function

The Triplet Loss Function is widely used in Image Similarity Problems. The data is arranged into triplets of images: Positive, Negative, and Anchor. The triplet loss function is a loss function that compares a reference input (called the anchor) to a matching input (called positive) and a non-matching input (called negative). The distance between the anchor and the positive input is minimized, while the distance between the anchor and the negative input is maximized. The triplet loss function outperforms other loss-based models.

---

## Applications of Siamese Neural Network:

1. Signature Verification
2. Face Recognition
3. Document Similarity
4. Document Authentication
5. Product Retrieval

---

## Tech Stack:

- Language Used: `Python`
- Packages used: `TensorFlow`, `Keras`, `Numpy`, `etc`

---

## Approach:

1. **Setting up the environment:**
   - Importing the necessary libraries and modules needed for the project.

2. **Understanding Dataset:**
   - Three different sets of documents: Driving license, social security number, and other documents.
   - 600 images in the training dataset, 150 images in the test dataset.
   - Set different parameters like brightness, saturation, contrast, etc., needed for data augmentation.

1. **Load and prepare the data**

2. **Build the model**

3. **Train the model**

4. **Test the model and make predictions**

5. **Draw Conclusions**

---

## Key Concepts Explored:

- Convolutional Neural Network
- Siamese Neural Network
- Why CNN is preferred in Siamese Neural Networks?
- Application of Siamese Neural Network
- Understanding image similarity
- Triplet Loss Function
- Data Augmentation
- Euclidean Distance
- Build Siamese Network with TensorFlow
- Predict Similar images using the trained model


---


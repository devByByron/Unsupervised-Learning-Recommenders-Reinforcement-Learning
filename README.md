# Unsupervised-Learning-Recommenders-Reinforcement-Learning

# 📘 Unsupervised Learning: Key Concepts

This guide summarizes the **Unsupervised Learning** section from the Coursera course  
**"Unsupervised Learning, Recommenders, Reinforcement Learning"**.  
It is designed to help beginners quickly grasp the main ideas.

---

## 🌱 What is Unsupervised Learning?
- Unlike **supervised learning** (where data has labels), unsupervised learning only has **raw inputs**.
- Goal: Find **hidden structure** in data.
- Two main techniques covered here:
  1. **Clustering (K-means)**
  2. **Anomaly Detection**

---

## 🔹 Clustering with K-means

### What is clustering?
- Group data points into **clusters** such that points in the same group are more similar to each other than to points in other groups.
- Example: Segmenting customers into "types" based on purchase behavior.

### K-means intuition
- Each cluster is represented by a **centroid**.
- Each data point is assigned to the **nearest centroid**.

### K-means algorithm
1. Choose the number of clusters, **K**.
2. Randomly initialize **K centroids**.
3. Repeat until convergence:
   - **Assignment step**: Assign each point to its nearest centroid.
   - **Update step**: Move centroids to the mean of assigned points.

### Optimization objective
- K-means minimizes the **distortion (cost)**:
 <img width="495" height="143" alt="image" src="https://github.com/user-attachments/assets/ecb73861-1100-4f79-aacf-feec0d8ab8d2" />

- Intuition: Make clusters **tight** and **well-separated**.

### Initializing K-means
- Poor initialization → bad local minima.
- Fix: Run algorithm multiple times with different initializations, keep the one with the **lowest cost**.

### Choosing the number of clusters
- **Elbow method**: Plot cost vs. K.
- Choose K at the “elbow” point where cost reduction slows down.

---

## 🔹 Anomaly Detection

### Finding unusual events
- Goal: Detect rare/unexpected examples.
- Applications:
  - Fraud detection
  - Engine failure
  - Network intrusion

### Gaussian (normal) distribution
- **Bell curve** distribution defined by:
  - Mean (\(\mu\))
  - Variance (\(\sigma^2\))
- Probability density function:
  \[
  p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
  \]

### Anomaly detection algorithm
1. **Fit Gaussian** distribution for each feature (\(\mu, \sigma^2\)).
2. Compute probability of a new example:
   \[
   p(x) = \prod_j p(x_j; \mu_j, \sigma_j^2)
   \]
3. If \(p(x) < \epsilon\) (threshold) → classify as **anomaly**.

### Developing & evaluating system
- Use a **validation set** with known anomalies.
- Evaluate using:
  - **Precision** = TP / (TP + FP)  
  - **Recall** = TP / (TP + FN)  
  - **F1 score** = \( \frac{2 \cdot \text{prec} \cdot \text{rec}}{\text{prec} + \text{rec}} \)

### Anomaly detection vs. supervised learning
- Use **supervised learning** if:
  - You have many examples of both normal and anomalous cases.
- Use **anomaly detection** if:
  - Anomalies are **rare**.
  - Few or no examples of anomalies exist.

### Choosing features
- Select features that capture **signals of abnormality**.
- Example: For engine monitoring, features could be:
  - Temperature
  - Vibration
  - Pressure
- Nonlinear features (e.g., squared terms) can improve detection.

---

## 🔑 Key Takeaways
- **Clustering (K-means)**: Groups data without labels.
- **Anomaly detection**: Identifies rare/unusual events using probabilities.
- **Gaussian distribution**: Mathematical backbone of anomaly detection.
- **Evaluation**: Use F1 score due to class imbalance.
- **Features matter**: Good feature design is crucial.

---



**Types of Unsupervised Learning algorithms**
- K-means
- Anomaly detection






## 🌱 What are Recommenders?
- Systems that **suggest items** (movies, music, products) to users.
- Goal: Help users discover content they’ll likely enjoy.

Two main approaches:
1. **Content-based filtering**
2. **Collaborative filtering**

---

## 🔹 Content-based Filtering
- Uses **item features** (e.g., genre, length, actors).
- Learns a user’s preferences based on items they’ve rated/liked.
- Recommends new items with **similar features**.

---

## 🔹 Collaborative Filtering
- Uses **user rating patterns** (no item features needed).
- Idea: “Users who are similar will like similar things.”
- **Algorithm**:
  - Each user has parameters \( W_j \).
  - Each item has features \( X_i \).
  - Predicted rating:
    \[
    \hat{y}_{i,j} = W_j \cdot X_i + b_j
    \]
  - Learn parameters by minimizing squared error.

---

## 🔹 Key Concepts

### Binary labels
- Many platforms don’t use 1–5 star ratings.
- Instead: **clicks, likes, favorites** (binary signals).
- Recommender predicts probability of positive interaction.

### Mean normalization
- Problem: Users rate differently (some high, some low).
- Fix: Subtract each user’s **average rating** before training.

### Finding related items
- Compute similarity between item vectors \(X_i\).
- Basis for **“Users who liked this also liked…”**.

### Collaborative vs Content-based
| Content-based | Collaborative |
|---------------|---------------|
| Needs item features | Needs lots of ratings |
| Works well with new users | Learns hidden patterns |
| Personalizes via features | Finds community-wide trends |

➡️ Real systems often **combine both (hybrid systems)**.

### Deep learning for recommendations
- Neural networks can process complex features:
  - Images, audio, text.
- Used by companies like Netflix and YouTube.

### Large catalogues
- Millions of items → too slow to check all.
- Solutions:
  - **Approximate nearest neighbors (ANN)** for fast search.
  - Pre-filter by popularity, recency, or category.

### Ethical use
- Dangers:
  - **Filter bubbles**: only seeing one viewpoint.
  - **Addiction/overuse**: optimized for engagement.
  - **Bias/fairness**: unequal recommendations.
- Good recommender design balances **accuracy, fairness, and well-being**.

---

## 🔹 Practical Tools

### TensorFlow Implementations
- Collaborative filtering: optimize parameters \(X, W, b\).
- Content-based filtering: neural networks on item features.

### PCA (optional)
- **Principal Component Analysis** reduces features while keeping variance.
- Benefits:
  - Faster training
  - Less noise

---

## 🔑 Takeaways
1. **Content-based filtering** = uses item features.  
2. **Collaborative filtering** = uses rating patterns.  
3. **Mean normalization** fixes rating bias.  
4. **Binary feedback** is powerful (likes, clicks).  
5. **Hybrid systems** combine both methods.  
6. **Deep learning** boosts personalization.  
7. **Ethics matter**: avoid bias and unhealthy engagement.  

---

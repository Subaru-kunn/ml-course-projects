# Regression (Linear and Logistic) and LDA using TensorFlow

This project was made as part of machine learning coursework. I have attached the problem statement along with the dataset used and codes for this project.

## 📋 Assignment Overview
This project aims to predict the Chance of Admit for graduate applicants based on various features such as GRE score, TOEFL score, CGPA, SOP, LOR and research experience. The notebook implements multiple machine learning models including **Linear Regression**, **Logistic Regression** and **Linear Discriminant Analysis (LDA)** using Python libraries such as NumPy, Pandas, Scikit-learn and TensorFlow.

Given data on university applicants, predict the **probability (Chance of Admit)** that a student will be accepted into a graduate program.

### Features include:
- GRE Score
- TOEFL Score
- University Rating
- SOP (Statement of Purpose) strength
- LOR (Letter of Recommendation) strength
- CGPA
- Research Experience (0 or 1)

### Target:
- `Chance of Admit` (between 0 and 1)

## Models Used

1. **Linear Discriminant Analysis (LDA)**  
   - Used for dimensionality reduction before regression.

2. **Linear Regression**  
   - For continuous prediction of `Chance of Admit`.

3. **Logistic Regression**  
   - For binary classification (e.g., admit vs not admit based on threshold).


# Medical Dataset Analysis with Genetic Algorithm and Particle Swarm Optimization

## Dataset Overview

Our project utilizes a medical dataset containing the following biomarkers:

1. **glucose_level**: Blood glucose concentration (normalized to 0-1)
   - Real-world range: 70-300 mg/dL
   - Normalized values: 0.0 (70 mg/dL) to 1.0 (300 mg/dL)
   - Clinical significance: Values >0.7 (>180 mg/dL) indicate hyperglycemia

2. **blood_pressure**: Normalized systolic blood pressure readings
   - Real-world range: 90-200 mmHg
   - Normalized values: 0.0 (90 mmHg) to 1.0 (200 mmHg)
   - Clinical significance: Values >0.75 (>160 mmHg) indicate hypertension

3. **bmi**: Body Mass Index
   - Real-world range: 15-45 kg/m²
   - Normalized values: 0.0 (15 kg/m²) to 1.0 (45 kg/m²)
   - Clinical significance: Values >0.67 (>35 kg/m²) indicate severe obesity

4. **cholesterol**: Normalized total cholesterol levels
   - Real-world range: 120-320 mg/dL
   - Normalized values: 0.0 (120 mg/dL) to 1.0 (320 mg/dL)
   - Clinical significance: Values >0.65 (>250 mg/dL) indicate hypercholesterolemia

5. **heart_rate**: Normalized heart rate measurements
   - Real-world range: 40-140 bpm
   - Normalized values: 0.0 (40 bpm) to 1.0 (140 bpm)
   - Clinical significance: Values >0.75 (>115 bpm) indicate tachycardia

6. **white_blood_cells**: Normalized white blood cell count
   - Real-world range: 3,000-20,000 cells/μL
   - Normalized values: 0.0 (3,000 cells/μL) to 1.0 (20,000 cells/μL)
   - Clinical significance: Values >0.65 (>14,000 cells/μL) indicate infection/inflammation

7. **red_blood_cells**: Normalized red blood cell count
   - Real-world range: 3.0-6.5 million cells/μL
   - Normalized values: 0.0 (3.0 million cells/μL) to 1.0 (6.5 million cells/μL)
   - Clinical significance: Values <0.3 (<4.0 million cells/μL) indicate anemia

8. **oxygen_saturation**: Blood oxygen levels
   - Real-world range: 70-100%
   - Normalized values: 0.0 (70%) to 1.0 (100%)
   - Clinical significance: Values <0.9 (<93%) indicate hypoxemia

9. **creatinine_level**: Kidney function marker
   - Real-world range: 0.5-5.0 mg/dL
   - Normalized values: 0.0 (0.5 mg/dL) to 1.0 (5.0 mg/dL)
   - Clinical significance: Values >0.35 (>1.3 mg/dL) indicate kidney dysfunction

10. **albumin_level**: Protein level indicator for liver function
    - Real-world range: 2.0-6.0 g/dL
    - Normalized values: 0.0 (2.0 g/dL) to 1.0 (6.0 g/dL)
    - Clinical significance: Values <0.25 (<3.0 g/dL) indicate malnutrition or liver disease

The target variable **disease_risk** is binary (0 = low risk, 1 = high risk), indicating whether a patient is at risk for developing cardiovascular disease based on these biomarkers.

## Disease Risk Calculation

### Mathematical Model for Disease Risk Prediction

In real-world medical practice, disease risk is calculated using complex multivariate models that consider the interactions between various biomarkers. Our neural network model aims to approximate these complex relationships. Here's how the disease risk calculation works in our system:

#### 1. Traditional Medical Risk Calculation (Simplified)

In traditional medical practice, cardiovascular disease risk might be calculated using established algorithms like the Framingham Risk Score or ASCVD Risk Calculator, which use weighted combinations of risk factors. A simplified version might look like:

```
Risk Score = (w₁ × glucose_level) + (w₂ × blood_pressure) + (w₃ × bmi) + 
             (w₄ × cholesterol) + (w₅ × heart_rate) + ... + constant
```

Where w₁, w₂, etc. are weights determined through epidemiological studies.

#### 2. Our Neural Network Approach

Our neural network creates a more sophisticated model by:

1. **Learning Non-linear Relationships**: Unlike traditional linear models, our neural network can capture complex non-linear relationships between biomarkers.

2. **Discovering Feature Interactions**: The hidden layers can identify interactions between features (e.g., how glucose and blood pressure together might amplify risk).

3. **Adaptive Weighting**: The network learns optimal weights for each biomarker based on the training data.

The final disease risk probability is calculated through:

```
p(disease_risk) = sigmoid(neural_network_output)
```

Where the neural network output is the weighted sum of activations from the final hidden layer, and the sigmoid function converts this to a probability between 0 and 1. If this probability exceeds 0.5, the patient is classified as high risk (1); otherwise, they're classified as low risk (0).

#### 3. Key Risk Patterns

Based on medical literature, our model is expected to identify these key risk patterns:

- **Metabolic Syndrome Pattern**: High glucose (>0.7) + High BMI (>0.67) + High blood pressure (>0.75) + High cholesterol (>0.65) strongly indicates disease risk
- **Cardiac Stress Pattern**: Elevated heart rate (>0.6) + Low oxygen saturation (<0.9) + High blood pressure (>0.75) indicates cardiovascular strain
- **Inflammatory Response**: Elevated white blood cells (>0.65) + High creatinine (>0.35) + Low albumin (<0.25) indicates systemic inflammation and organ stress

#### 4. Risk Thresholds

The binary classification (0 or 1) is determined by whether the calculated risk probability exceeds the threshold of 0.5, but the underlying probability provides a more nuanced view of risk:

- 0.0-0.2: Very low risk
- 0.2-0.4: Low risk
- 0.4-0.6: Moderate risk
- 0.6-0.8: High risk
- 0.8-1.0: Very high risk

## Project Objectives

Our project applies two advanced optimization algorithms - Genetic Algorithm (GA) and Particle Swarm Optimization (PSO) - to perform two critical tasks in medical predictive modeling:

1. **Neural Network Hyperparameter Tuning**: Optimizing the architecture and learning parameters of neural networks to achieve maximum predictive accuracy for disease risk assessment.

2. **Feature Selection**: Identifying the most significant biomarkers that contribute to disease risk prediction, potentially reducing the number of tests needed for accurate diagnosis.

## Algorithm Implementation

### Genetic Algorithm (GA)

The Genetic Algorithm in our project mimics natural selection to evolve optimal solutions:

1. **Chromosome Representation**:
   - For hyperparameter tuning: Each chromosome encodes neural network parameters (hidden layers, learning rate, activation function, batch size, dropout rate)
   - For feature selection: Each gene represents whether a specific biomarker is included (1) or excluded (0)

2. **Evolutionary Process**:
   - **Selection**: Chromosomes with higher fitness (better prediction accuracy) have higher chances of being selected
   - **Crossover**: Combines hyperparameter configurations from two parent solutions
   - **Mutation**: Introduces random variations to explore new hyperparameter combinations

3. **Medical Application**:
   - Identifies optimal neural network architectures for disease risk prediction
   - Discovers which combination of biomarkers yields the highest diagnostic accuracy

### Particle Swarm Optimization (PSO)

PSO simulates social behavior where particles (potential solutions) move through the search space:

1. **Particle Representation**:
   - For hyperparameter tuning: Each particle's position represents a specific neural network configuration
   - For feature selection: Each dimension represents the importance weight of a biomarker

2. **Optimization Process**:
   - **Personal Best**: Each particle remembers its best position (highest accuracy)
   - **Global Best**: All particles are influenced by the best solution found by any particle
   - **Velocity Update**: Particles adjust their movement based on personal and global knowledge

3. **Medical Application**:
   - Efficiently searches for optimal neural network parameters for disease prediction
   - Identifies optimal feature subsets by converging on the most predictive biomarkers

## Practical Benefits

1. **Improved Diagnostic Accuracy**: By optimizing neural network models, we can achieve higher accuracy in predicting disease risk from biomarker data.

2. **Cost-Effective Screening**: Feature selection identifies which tests (biomarkers) are most critical, potentially reducing unnecessary medical tests.

3. **Personalized Medicine**: Optimized models can better capture complex relationships between biomarkers, supporting more personalized risk assessment.

4. **Early Intervention**: More accurate predictive models enable earlier identification of high-risk patients, allowing for preventive interventions.

5. **Algorithm Comparison**: By comparing GA and PSO performance, we can determine which optimization approach is more effective for medical predictive modeling.

## Technical Implementation

Our implementation processes the medical dataset through these key steps:

1. **Data Preprocessing**: Splitting data into training, validation, and test sets while maintaining class distribution.

2. **Neural Network Construction**: Building a PyTorch neural network model with configurable architecture.

3. **Optimization Process**:
   - GA evolves a population of neural network configurations over multiple generations
   - PSO moves particles (potential configurations) through the hyperparameter space
   - Both algorithms use validation accuracy as the fitness/objective function

4. **Performance Evaluation**: Testing the optimized models on unseen data to measure generalization ability.

5. **Visualization**: Generating convergence plots, performance comparisons, and feature importance visualizations.

## Conclusion

This project demonstrates how evolutionary and swarm intelligence algorithms can be applied to medical data analysis, potentially improving disease risk prediction while identifying the most relevant biomarkers. The comparison between GA and PSO provides insights into which algorithm performs better for medical predictive modeling tasks.

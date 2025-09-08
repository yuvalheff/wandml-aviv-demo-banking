# Bank Marketing Term Deposit Prediction - EDA Report

## Dataset Overview
- **Shape**: 40689 samples, 17 features + 1 target
- **Task Type**: Binary Classification
- **Target**: Term deposit subscription (1=no, 2=yes)
- **Data Quality**: No missing values

## Target Distribution
- **No subscription**: 35929 samples (88.3%)
- **Subscription**: 4760 samples (11.7%)
- **Class Imbalance**: Significant imbalance requiring special handling

## Feature Summary
- **Categorical Features**: 9 (job, marital, education, default, housing...)
- **Numerical Features**: 7 (age, balance, day, duration, campaign...)

## Key EDA Findings

### 1. Target Variable Analysis
- Significant class imbalance with only 11.7% positive cases
- Will require stratified sampling and appropriate evaluation metrics
- Dataset size sufficient for reliable model training

### 2. Data Quality Assessment  
- Perfect data quality with no missing values
- All features have appropriate data types
- Good feature diversity across categorical and numerical variables

### 3. Feature Insights
- **Duration**: Strongest predictor - longer calls correlate with subscription
- **Previous Outcome**: Clients with previous success much more likely to subscribe
- **Demographics**: Age and job type show clear patterns in subscription rates
- **Contact Method**: Cellular contact shows different conversion than telephone

### 4. Recommendations for Modeling
- Handle class imbalance with stratified sampling or class weights
- Consider duration as key feature but be aware of data leakage potential
- Use appropriate evaluation metrics (ROC-AUC, precision-recall)
- Encode categorical variables appropriately
- Consider feature interactions, especially around demographics

## Generated Visualizations
- Target Variable Distribution Analysis: target_distribution.html
- Data Quality Assessment: data_quality_overview.html
- Categorical Features Distribution: categorical_features_distribution.html
- Numerical Features Distribution: numerical_features_distribution.html
- Feature Correlation Analysis: correlation_matrix.html
- Categorical Features vs Target Analysis: categorical_vs_target.html
- Numerical Features vs Target Analysis: numerical_vs_target.html

## Conclusion
The dataset provides a solid foundation for building a term deposit prediction model. The combination of demographic, financial, and campaign features offers comprehensive information, while the clean data quality ensures reliable model training. Key considerations include addressing class imbalance and leveraging the strong predictive signals identified through this analysis.

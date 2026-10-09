# Flood Probability Prediction Using Linear Regression

## Project Overview
This project uses machine learning to estimate the probability of flooding from environmental and infrastructure-related input features. It is implemented in a Jupyter Notebook using Python and Scikit-learn.

The notebook trains a **Linear Regression** model to predict the `FloodProbability` target.

## Objectives
- Load and inspect the training dataset.
- Check missing values and review dataset information.
- Explore numeric relationships using a correlation heatmap.
- Visualize feature distributions with histograms and box plots.
- Detect and cap outliers using the Interquartile Range (IQR) method.
- Train and evaluate a Linear Regression model.
- Save and reload the trained model.
- Generate a prediction for a sample input.

## Dataset
The notebook expects a file named `train.csv.zip` at `/content/train.csv.zip` and a target column named `FloodProbability`.

The dataset file is not included in this repository. Place the dataset in your environment and update the CSV path if needed.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook / Google Colab

## Workflow
1. **Load data:** Read `train.csv.zip` into a Pandas DataFrame.
2. **Inspect data:** View the DataFrame, check null values, and review data types.
3. **Exploratory data analysis:** Plot a correlation heatmap, histograms, and box plots.
4. **Outlier handling:** Calculate IQR bounds and cap numeric values. The notebook caps feature outliers before model training.
5. **Prepare features and target:** Use all columns except `FloodProbability` as `X`, and `FloodProbability` as `y`. The `id` column is removed.
6. **Split data:** Use an 80/20 train-test split with `random_state=42`.
7. **Train model:** Fit a `LinearRegression` estimator.
8. **Evaluate:** Calculate MAE, MSE, RMSE, and R² score.
9. **Save and load:** Save the model as `flood_model.joblib`, then load it for predictions.

## Installation
Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

## How to Run
1. Clone or download this project.
2. Make sure the dataset file `train.csv.zip` is available.
3. Open `Regression_with_a_Flood_Prediction_Dataset.ipynb` in Jupyter Notebook or Google Colab.
4. Update the dataset path in the first code cell if necessary.
5. Run the notebook cells in order.
6. Check the printed regression metrics and sample prediction.

## Model Evaluation
The notebook reports these metrics:
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **MSE (Mean Squared Error):** Average squared prediction error.
- **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target's units.
- **R² Score:** Measures how well the model explains variation in the target; a higher value generally indicates a better fit.

Actual metric values depend on the dataset and execution environment.

## Saved Model
After training, the notebook saves the fitted model to:

```text
flood_model.joblib
```

Load it with:

```python
import joblib

model = joblib.load("flood_model.joblib")
```

## Sample Prediction
The notebook includes a sample input with 20 feature values. Make sure prediction inputs contain the same features in the same order as the training data.

```python
new_data = [[
    5, 6, 4, 3, 7,
    5, 6, 4, 5, 3,
    4, 5, 6, 3, 4,
    5, 7, 4, 3, 2
]]

prediction = model.predict(new_data)
print("Flood Probability:", prediction[0])
```

**Note:** This is a sample demonstration input. It is not a validated real-world flood forecast. Model outputs should not be used as the sole basis for safety-critical decisions.

## Important Notes / Future Improvements
- Split the data into training and test sets **before** fitting outlier caps, so preprocessing is learned only from the training data and evaluation leakage is reduced.
- The notebook currently performs outlier capping across numeric columns before the train-test split. In a more robust pipeline, fit preprocessing on training data only.
- Confirm the dataset feature order before making predictions. Prefer a Pandas DataFrame with explicit feature names instead of a raw list.
- Compare Linear Regression with models such as Random Forest Regressor, Extra Trees Regressor, or Gradient Boosting.
- Consider cross-validation and residual plots for more reliable model assessment.

## Project Structure
```text
.
├── Regression_with_a_Flood_Prediction_Dataset.ipynb
├── train.csv.zip                 # Dataset (add separately)
├── flood_model.joblib            # Created after running the notebook
└── README.md
```

## Author
Add your name and project links here.

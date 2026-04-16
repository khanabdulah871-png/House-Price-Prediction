# House Price Prediction

A machine learning project for predicting house prices using **Linear Regression** on the **Housing.csv** dataset.  
This notebook performs **EDA, preprocessing, feature engineering, feature selection (RFE), model training, and evaluation**.

---

## 📌 Project Overview

This project builds a **house price prediction model** using a regression approach.  
The workflow includes:

- Loading and exploring the housing dataset
- Visualizing numerical and categorical features
- Encoding categorical variables
- Scaling numerical variables
- Selecting important features using **RFE (Recursive Feature Elimination)**
- Training a **Linear Regression** model
- Evaluating the model using:
  - Residual distribution
  - `y_test vs y_pred` scatter plot
  - OLS summary statistics

---

## 📂 Project Structure

```bash
House_Price_Prediction/
│── House_Price_Prediction.ipynb   # Main notebook
│── Housing.csv                    # Dataset file
│── README.md                      # Project documentation
```

---

## 🧰 Libraries Used

The project uses the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels`

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
```

---

## 📊 Dataset

The dataset used is **Housing.csv**.

### Expected columns:
- `price` (target variable)
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

> Make sure `Housing.csv` is placed in the correct path before running the notebook.

---

## 🔍 Workflow

### 1. Import Libraries
The notebook starts by importing required libraries for:
- Data handling
- Visualization
- Model building
- Statistical analysis

### 2. Load Dataset
The dataset is loaded using:

```python
df = pd.read_csv('Housing.csv')
```

> In your notebook, the dataset path is currently set to Google Drive:
```python
df = pd.read_csv('/content/drive/MyDrive/Housing.csv')
```
If you are running locally, replace it with:
```python
df = pd.read_csv('Housing.csv')
```

### 3. Exploratory Data Analysis (EDA)
The notebook performs:
- `head()`
- `info()`
- `describe()`
- Missing value check
- Duplicate check

### 4. Data Visualization
Visualizations include:
- Distribution plot of house prices
- Boxplot of house prices
- Frequency plots for categorical variables
- Boxplots of categorical features vs price
- Scatter plots of numerical features vs price
- Pairplot
- Correlation heatmap

### 5. Encoding Categorical Variables
Categorical columns are converted into dummy variables using:

```python
pd.get_dummies(..., drop_first=True)
```

This avoids the dummy variable trap and prepares the data for regression.

### 6. Train-Test Split
The data is split into:
- **75% Training**
- **25% Testing**

```python
train_test_split(df, train_size=0.75, test_size=0.25, random_state=100)
```

### 7. Feature Scaling
Numerical features are scaled using **MinMaxScaler**:

```python
scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])
```

### 8. Feature Selection (RFE)
The notebook uses **Recursive Feature Elimination (RFE)** with **LinearRegression** to select the top 10 features.

```python
rfe = RFE(estimator=LinearRegression(), n_features_to_select=10)
```

### 9. Model Building
A regression model is built using **statsmodels OLS**:

```python
lm = sm.OLS(y_train, X_train_new).fit()
```

The notebook also checks:
- **p-values**
- **Adjusted R²**
- **VIF (Variance Inflation Factor)**

Features are removed iteratively (e.g., `bedrooms`, `yes`) based on significance and multicollinearity.

### 10. Model Evaluation
The model is evaluated using:

#### Residual Distribution
- Error terms should be centered around zero
- Approximate normal distribution indicates a good linear model

#### `y_test vs y_pred` Scatter Plot
- Shows how close predictions are to actual values
- The closer points are to the 45° line, the better the model

#### OLS Summary
- Includes:
  - R-squared
  - Adjusted R-squared
  - p-values
  - F-statistic
  - confidence intervals

---

## ▶️ How to Run

### Option 1: Jupyter Notebook
```bash
jupyter notebook
```
Then open:
- `House_Price_Prediction.ipynb`

### Option 2: Google Colab
- Upload the notebook to Colab
- Mount Google Drive if using the original file path
- Ensure `Housing.csv` is available in Drive

---

## 📈 Expected Output

After running the notebook, you will get:

- EDA summaries
- Visual analysis plots
- Selected features from RFE
- Final OLS regression summary
- Residual error distribution graph
- `y_test vs y_pred` scatter plot

---

## ⚠️ Important Improvement (Recommended Fix)

In the notebook, the test set scaling is currently done as:

```python
df_test[numerical_list] = scaler.fit_transform(df_test[numerical_list])
```

This should be changed to:

```python
df_test[numerical_list] = scaler.transform(df_test[numerical_list])
```

### Why?
- `fit_transform()` on test data causes **data leakage / inconsistent scaling**
- Always `fit` on training data, then only `transform` test data

✅ Correct workflow:
```python
scaler = MinMaxScaler()
df_train[numerical_list] = scaler.fit_transform(df_train[numerical_list])
df_test[numerical_list] = scaler.transform(df_test[numerical_list])
```

---

## 🚀 Future Improvements

You can improve this project by:

- Adding evaluation metrics:
  - `R² Score`
  - `MAE`
  - `MSE`
  - `RMSE`
- Comparing multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest Regressor
- Saving the trained model with `joblib` or `pickle`
- Creating a web app using **Streamlit** or **Flask**

---

## 📝 Sample Conclusion

The project successfully builds a **house price prediction model** using linear regression.  
The residuals are approximately centered around zero, and the model captures the general trend between actual and predicted values. While the model shows reasonable predictive ability, further improvement can be achieved by fixing test-set scaling, adding performance metrics, and experimenting with more advanced models.

---

## 👨‍💻 Author

Project prepared by **Muhammad_Abdullah_Khan**

---

## 📄 License

This project is for **educational and learning purposes**.

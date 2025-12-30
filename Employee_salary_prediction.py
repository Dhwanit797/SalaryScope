import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------
# Load dataset
# -----------------------------------

df = pd.read_csv("multi_feature_salary.csv")

X = df[["YearsExperience", "Age", "EducationLevel"]]
y = df["Salary"]

# -----------------------------------
# Train-test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Feature scaling
# -----------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# Model training
# -----------------------------------

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# -----------------------------------
# Predictions for evaluation
# -----------------------------------

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# -----------------------------------
# Menu-driven program
# -----------------------------------

while True:
    print("\n==============================")
    print(" EMPLOYEE SALARY PREDICTOR ")
    print("==============================")
    print("1. Predict salary for new employee")
    print("2. View model accuracy (MAE & MSE)")
    print("3. View feature importance")
    print("4. Visualize data")
    print("5. Exit")

    choice = input("Enter your choice: ")

    # -----------------------------
    # Predict new employee salary
    # -----------------------------
    if choice == "1":
        try:
            exp = float(input("Enter years of experience: "))
            age = float(input("Enter age: "))
            edu = int(input("Education level (1-HS, 2-Bachelor, 3-Master, 4-PhD): "))

            new_employee = [[exp, age, edu]]
            new_employee_scaled = scaler.transform(new_employee)

            salary_pred = model.predict(new_employee_scaled)
            print(f"\nPredicted Salary: {salary_pred[0]:.2f}")

        except ValueError:
            print("Invalid input. Please enter numeric values.")

    # -----------------------------
    # Model evaluation
    # -----------------------------
    elif choice == "2":
        print("\nModel Performance:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print("Interpretation: On average, predictions are off by MAE amount.")

    # -----------------------------
    # Feature importance
    # -----------------------------
    elif choice == "3":
        print("\nFeature Importance (Coefficients):")
        for feature, coef in zip(X.columns, model.coef_):
            print(f"{feature}: {coef:.2f}")

        print("\nHigher absolute value = stronger impact on salary.")

    # -----------------------------
    # Visualization
    # -----------------------------
    elif choice == "4":
        plt.scatter(df["YearsExperience"], df["Salary"])
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.title("Salary vs Experience")
        plt.show()

    # -----------------------------
    # Exit
    # -----------------------------
    elif choice == "5":
        print("Exiting program...")
        break

    else:
        print("Invalid option. Please try again.")

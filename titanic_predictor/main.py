import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select relevant features
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical 'Sex' column to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2f}")

# User input for prediction
print("\n--- Predict Your Survival ---")
try:
    pclass = int(input("Enter your passenger class (1, 2, or 3): "))
    sex_input = input("Enter your sex (male or female): ").lower()
    age = float(input("Enter your age: "))
    sibsp = int(input("Enter number of siblings/spouses aboard: "))
    parch = int(input("Enter number of parents/children aboard: "))
    fare = float(input("Enter your fare: "))

    # Convert sex input to numeric
    sex = 0 if sex_input == "male" else 1

    # Create feature vector and predict
    person = [[pclass, sex, age, sibsp, parch, fare]]
    result = model.predict(person)

    print("\nPrediction: You would", "survive 💚" if result[0] == 1 else "not survive 💀")

except Exception as e:
    print(f"\nInvalid input. Please enter the correct data types.\nError: {e}")

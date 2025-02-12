import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import csv

file_path = "Sorted_Exoplanets.csv"

# Initialize lists to hold the data from the CSV file
orbital_period = []
orbital_distance = []
host_star_type = []
planet_type = []
size = []
mass = []
gravity = []  # gravity will be calculated later

try:
    with open(file_path, 'r') as file:
        content = csv.reader(file)
        next(content)  # Skip header if present
        for line in content:
            orbital_period.append(float(line[1]))
            orbital_distance.append(float(line[2]))
            host_star_type.append(line[3])
            size.append(float(line[5]))  # Assuming size is in Earth radii
            gravity.append(float(line[7]))  # Assuming gravity is in m/s^2

except FileNotFoundError:
    print("Error: Sorted_Exoplanets.csv was not found")
    exit()
except PermissionError:
    print("Error: No Permissions to read Sorted_Exoplanets.csv")
    exit()


orbital_period.pop(0)
orbital_distance.pop(0)
host_star_type.pop(0)
size.pop(0)
gravity.pop(0)
# Prepare data for model
df = pd.DataFrame({
    'orbital_period': orbital_period,
    'orbital_distance': orbital_distance,
    'host_star_type': host_star_type,
    'size': size,
    'gravity': gravity
})

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Define features and target variables
X = df.drop(columns=['size', 'gravity'])
y = df[['orbital_period', 'orbital_distance', 'size', 'gravity']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a MultiOutputRegressor with RandomForestRegressor
multi_output_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
multi_output_model.fit(X_train_scaled, y_train)

# Function to predict characteristics of a planet based on user input
def predict_planet_characteristics(user_input):
    # Prepare user input (convert to DataFrame and apply the same scaling and encoding as the training data)
    user_input_df = pd.DataFrame([user_input])

    # One-hot encode categorical columns (host_star_type)
    user_input_df = pd.get_dummies(user_input_df, drop_first=True)

    # Make sure the input DataFrame has the same columns as the model's training data
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)

    # Scale the input features (same scaling used during training)
    user_input_scaled = scaler.transform(user_input_df)

    # Predict the characteristics (orbital_period, orbital_distance, size, gravity)
    prediction = multi_output_model.predict(user_input_scaled)

    return prediction[0]

# User input example (partial data)
orbital_period = float(input("Enter an Orbital Period (Days): "))
orbital_distance = float(input("Enter an Orbital Distance (AU): "))
star_type = input("Enter a star type (G or K): ")

g = 0
k = 0
if star_type == "G":
    g = 1
elif star_type == "K":
    k = 1

# Define user input in the format expected by the model
user_input_example = {
    'orbital_period': orbital_period,  # days
    'orbital_distance': orbital_distance,  # AU
    'host_star_type_G': g,  # one-hot encoded value for G-type
    'host_star_type_K': k,  # one-hot encoded value for K-type
}

# Make a prediction for the given user input
predicted_characteristics = predict_planet_characteristics(user_input_example)

# Display the predicted characteristics
print(f"Predicted Characteristics for the Planet:")
print(f"Orbital Period: {predicted_characteristics[0]} days")
print(f"Orbital Distance: {predicted_characteristics[1]} AU")
print(f"Size: {predicted_characteristics[2]} Earth radii")
print(f"Gravity: {predicted_characteristics[3]} m/s^2")

input("Press Anything to Exit")

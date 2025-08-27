import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the disease-symptom dataset
df = pd.read_csv("Disease_Symptom.csv")  # Make sure this file exists in your folder

# Fill missing symptom values with "None"
df.fillna("None", inplace=True)

# Get all symptom columns
symptom_columns = [col for col in df.columns if "Symptom" in col]

# Create a list of all unique symptoms
all_symptoms = set()
for col in symptom_columns:
    all_symptoms.update(df[col].unique())

all_symptoms.discard("None")  # remove empty values
all_symptoms = sorted(list(all_symptoms))  # final symptom list

# One-hot encode the symptoms for each row
def encode_symptoms(row):
    present_symptoms = set(row[symptom_columns])
    return [1 if symptom in present_symptoms else 0 for symptom in all_symptoms]

X = df.apply(encode_symptoms, axis=1, result_type='expand')
X.columns = all_symptoms

# Encode disease names
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model and data for later use
joblib.dump(model, "doctor_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(all_symptoms, "symptom_list.pkl")

print("✅ Model trained and saved successfully.")

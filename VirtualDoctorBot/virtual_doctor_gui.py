import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# Load model and data
model = joblib.load("doctor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptom_list = joblib.load("symptom_list.pkl")

# Load precaution dataset
precautions_df = pd.read_csv("Symptom_Precaution.csv")  # Make sure file is in same folder

# Create GUI window
window = tk.Tk()
window.title("Virtual Doctor Bot")
window.geometry("500x600")
window.configure(bg="#e0f7fa")

selected_symptoms = []

# Heading
tk.Label(window, text="Select your symptoms:", font=("Arial", 14, "bold"), bg="#e0f7fa").pack(pady=10)

# Scrollable list of symptoms
frame = tk.Frame(window)
frame.pack()

symptom_vars = {}

canvas = tk.Canvas(frame, height=300)
scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scroll_frame = tk.Frame(canvas)

scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

for symptom in symptom_list:
    var = tk.IntVar()
    cb = tk.Checkbutton(scroll_frame, text=symptom, variable=var, bg="#ffffff")
    cb.pack(anchor="w")
    symptom_vars[symptom] = var

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Predict function
def predict_disease():
    input_data = [1 if symptom_vars[sym].get() == 1 else 0 for sym in symptom_list]
    if sum(input_data) == 0:
        messagebox.showwarning("Input Missing", "Please select at least one symptom.")
        return
    
    prediction = model.predict([input_data])[0]
    disease_name = label_encoder.inverse_transform([prediction])[0]

    # Get precautions
    precautions = precautions_df[precautions_df["Disease"] == disease_name]
    if not precautions.empty:
        p = precautions.iloc[0][1:].dropna().values
        advice = "\n".join(f"- {step}" for step in p)
    else:
        advice = "No precaution data available."

    # Show result
    messagebox.showinfo("Prediction", f"🩺 Predicted Disease: {disease_name}\n\n📝 Precautions:\n{advice}")

# Button
tk.Button(window, text="Predict Disease", command=predict_disease, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(pady=20)

# Start GUI loop
window.mainloop()

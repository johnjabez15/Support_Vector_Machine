from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")

# Load the trained model pipeline.
# The pipeline includes the scaler and the SVM classifier.
try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please run train_model.py first.")
    exit()
    
@app.route("/")
def index():
    """Renders the main page with the user input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the form submission and returns the prediction result."""
    try:
        # Get the form data from the user input
        age = int(request.form["age"])
        time_on_site = float(request.form["time_on_site"])
        pages_viewed = int(request.form["pages_viewed"])
        added_to_cart = int(request.form["added_to_cart"])

        # Create a DataFrame from the input, ensuring column order matches the training data
        input_data = pd.DataFrame([[
            age,
            time_on_site,
            pages_viewed,
            added_to_cart
        ]], columns=["Age", "TimeOnSite", "PagesViewed", "AddedToCart"])
        
        # Use the pipeline to make a prediction.
        # It will automatically scale the input data before feeding it to the SVM model.
        prediction = pipeline.predict(input_data)[0]
        
        # Get the probability of the prediction
        prediction_proba = pipeline.predict_proba(input_data)[0]
        purchase_probability = prediction_proba[1] # Probability of a purchase (class 1)
        
        # Interpret the prediction
        result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
        
        return render_template("result.html", result=result, probability=f"{purchase_probability:.2f}")

    except Exception as e:
        # Handle potential errors during prediction
        return render_template("error.html", error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)


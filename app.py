from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

print("Flask app starting...")

# Load the model (make sure loan_model.pkl exists in the same folder)
model = joblib.load("loan_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    # Default empty values for all inputs
    input_data = {
        'Age': '',
        'Experience': '',
        'Income': '',
        'Family': '',
        'CCAvg': '',
        'Education': '',
        'Mortgage': '',
        'Securities_Account': '',
        'CD_Account': '',
        'Online': '',
        'CreditCard': ''
    }
    result = None

    if request.method == "POST":
        # Read all inputs from form, convert to float or int
        for key in input_data.keys():
            val = request.form.get(key)
            input_data[key] = val  # Save as string for template value
        try:
            # Convert inputs to float/int in correct order
            features = [
                float(input_data['Age']),
                float(input_data['Experience']),
                float(input_data['Income']),
                float(input_data['Family']),
                float(input_data['CCAvg']),
                int(input_data['Education']),
                float(input_data['Mortgage']),
                int(input_data['Securities_Account']),
                int(input_data['CD_Account']),
                int(input_data['Online']),
                int(input_data['CreditCard']),
            ]
            features_np = np.array([features])
            prediction = model.predict(features_np)[0]

            if prediction == 1:
                result = "Loan Approved ✅"
            else:
                result = "Loan Not Approved ❌"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result, **input_data)

if __name__ == "__main__":
    app.run(debug=True)

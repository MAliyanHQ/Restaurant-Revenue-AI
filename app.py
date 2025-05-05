from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('linear_revenue_model.pkl')  # Load your trained linear regression model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and validate input
        seating_capacity = float(request.form['seating_capacity'])
        average_meal_price = float(request.form['average_meal_price'])

        if seating_capacity <= 0 or average_meal_price <= 0:
            return render_template('index.html', error="Inputs must be greater than 0.")

        # Prepare input for model
        input_data = np.array([[seating_capacity, average_meal_price]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Check for unrealistic negative prediction
        if prediction < 0:
            return render_template('index.html', error="Your input values may not be valid in the real-world scenario.")

        # Calculate monthly revenue
        monthly_revenue = prediction / 12

        return render_template('index.html',
                               prediction=f"Yearly Revenue: ${prediction:,.2f}",
                               monthly_revenue=f"Monthly Revenue: ${monthly_revenue:,.2f}")
    except ValueError:
        return render_template('index.html', error="Please enter valid numbers.")
    except Exception as e:
        return render_template('index.html', error=f"Unexpected error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

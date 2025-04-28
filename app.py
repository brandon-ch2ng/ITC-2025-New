from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML models
duration_model = joblib.load('models/outage_duration_model.pkl')
customers_model = joblib.load('models/customers_affected_model.pkl')

# Define options
circuit_names = ['Adams', 'Alabama', 'Blue Jay', 'Dinan', 'Gorilla', 'Grand', 'Green', 'Hoover', 'Jefferson', 'Johnson',
                 'Lightning', 'Lincoln', 'Logan', 'Magenta', 'Monterey', 'Orange', 'Oregon', 'Roosevelt', 'Thunder',
                 'Washington', 'Yellow']
outage_causes = ['Animal', 'OH Equipment Failure', 'Operation', 'Other', 'Third Party', 'UG Equipment Failure',
                 'Vegetation', 'Weather']
kv_levels = ['4', '12', '16']
regions = ['Coastal', 'Desert', 'Mountain', 'North']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        customer_count = float(request.form['customer_count'])
        circuit_miles = float(request.form['circuit_miles'])
        selected = {
            'circuit': request.form['circuit_name'],
            'cause': request.form['outage_cause'],
            'kv': request.form['kv_level'],
            'region': request.form['region'],
            'month': request.form['month'],
            'day': request.form['day_of_week']
        }

        # Initialize all binary features as 0
        features = [customer_count, circuit_miles]
        for name in circuit_names:
            features.append(1 if name == selected['circuit'] else 0)
        for cause in outage_causes:
            features.append(1 if cause == selected['cause'] else 0)
        for kv in kv_levels:
            features.append(1 if kv == selected['kv'] else 0)
        for region in regions:
            features.append(1 if region == selected['region'] else 0)
        for month in months:
            features.append(1 if month == selected['month'] else 0)
        for day in days_of_week:
            features.append(1 if day == selected['day'] else 0)

        # Predict
        duration = duration_model.predict([features])[0]
        customers = customers_model.predict([features])[0]
        cmi = duration * customers

        return render_template('result.html',
                               duration=round(duration, 2),
                               customers=int(customers),
                               cmi=int(cmi))

    # In case the method is GET, you should return a form page
    return render_template('prediction.html', circuit_names=circuit_names, outage_causes=outage_causes,
                           kv_levels=kv_levels, regions=regions, months=months, days=days_of_week)

if __name__ == '__main__':
    app.run(debug=True)

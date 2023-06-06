from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('Churn_Prediciton_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    gender = request.form.get('gender')
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    tenure = request.form.get('tenure')
    PhoneService = request.form.get('PhoneService')
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = int(request.form.get('DeviceProtection'))
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    Contract = request.form.get('Contract')
    PaperlessBilling = request.form.get('PaperlessBilling')
    PaymentMethod = request.form.get('PaymentMethod')
    MonthlyCharges = float(request.form.get('MonthlyCharges'))
    TotalCharges = float(request.form.get('TotalCharges'))

    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
    })

    # Make predictions using the pre-trained model
    prediction = model.predict(input_data)

    # Display the predicted churn category
    if prediction[0] == 0:
        churn_category = 'Churn'
    else:
        churn_category = 'No Churn'

    return render_template('result.html', prediction_text='Predicted Customer: {}'.format(churn_category))

if __name__ == '__main__':
    app.run(debug=True)

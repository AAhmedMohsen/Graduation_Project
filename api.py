from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained models
disease_model = joblib.load("random_forest.pkl")
heart_disease_model = joblib.load("HeartDisease.pkl")

# Load symptom severity data
df1 = pd.read_csv("Symptom-severity.csv")
df1['Symptom'] = df1['Symptom'].str.replace('_', ' ')

# Load disease description and precautions
descrip_for_symptom = pd.read_csv("symptom_Description.csv")
precaution_to_avoid_symptom = pd.read_csv("symptom_precaution.csv")

def predict_disease(model, symptoms):
    """
    Predict the disease based on user-provided symptoms.
    """
    X = np.array(df1["Symptom"])
    Y = np.array(df1["weight"])
    
    for j in range(len(symptoms)):
        if symptoms[j] != 0:
            for k in range(len(X)):
                if symptoms[j].lower() == X[k].lower():
                    symptoms[j] = Y[k]
                    break
            else:
                symptoms[j] = 0
    
    predict_symptoms = [symptoms]
    pred = model.predict(predict_symptoms)
    
    # Get disease details
    description = descrip_for_symptom[descrip_for_symptom['Disease'] == pred[0]].values[0][1]
    
    recommendations = precaution_to_avoid_symptom[precaution_to_avoid_symptom['Disease'] == pred[0]]
    c = np.where(precaution_to_avoid_symptom['Disease'] == pred[0])[0][0]
    precautions = [precaution_to_avoid_symptom.iloc[c, i] for i in range(1, len(precaution_to_avoid_symptom.iloc[c]))]
    
    return {
        "disease": pred[0],
        "description": description,
        "precautions": precautions
    }

@app.route("/predict_disease", methods=["POST"])
def predict_disease_route():
    try:
        data = request.get_json()
        symptoms_list = data.get("symptoms")
        
        if symptoms_list is None or not isinstance(symptoms_list, list):
            return jsonify({"error": "Missing or invalid 'symptoms' in request body"}), 400
        
        result = predict_disease(disease_model, symptoms_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_heart_disease", methods=["POST"])
def predict_heart_disease():
    try:
        data = request.json
        input_data = np.asarray(data['input']).reshape(1, -1)
        
        prediction = heart_disease_model.predict(input_data)
        result = "Your results are not normal :(" if prediction[0] == 1 else "Your results are normal :)"
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask('churn-prediction')

# logistic regression model
model_file_path = 'models/lr_model_churn_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))

# encoding model DictVectorizer
encoding_model_file_path = 'models/encoding_model.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

# Global variables for prediction
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender',
'seniorcitizen',
'partner',
'dependents',
'phoneservice',
'multiplelines',
'internetservice',
'onlinesecurity',
'onlinebackup',
'deviceprotection',
'techsupport',
'streamingtv',
'streamingmovies',
'contract',
'paperlessbilling',
'paymentmethod']
le_enc_cols = ['gender', 'partner', 'dependents','paperlessbilling', 'phoneservice']
gender_map = {'male': 0, 'female': 1}
y_n_map = {'yes': 1, 'no': 0}

treshold = 0.281


@app.route('/predict', methods=['POST'])
def predict():
    
    customer = request.get_json()

    df_input = pd.DataFrame(customer, index=[0])
    scaler = MinMaxScaler()

    df_input[numerical] = scaler.fit_transform(df_input[numerical])

    for col in le_enc_cols:
        if col == 'gender':
            df_input[col] = df_input[col].map(gender_map)
        else:
            df_input[col] = df_input[col].map(y_n_map)

    dicts_df = df_input[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    y_pred = model.predict_proba(X)[:, 1]
    churn_descision = (y_pred >= treshold)

    result = {'churn_probability': float(y_pred[0]), 'churn': bool(churn_descision[0])}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9697)

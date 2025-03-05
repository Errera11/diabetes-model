import json

import joblib
import numpy as np
from django.http import JsonResponse
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime

model = tf.keras.models.load_model('model.h5')
QT = joblib.load('quantile_transformer.pkl')
ST = joblib.load('standard_transformer.pkl')

def scaleVal(val, scaler_type='standard'):
    if scaler_type == 'standard':
        scaledVal = ST.transform(np.array(val).reshape(1, -1))
    elif scaler_type == 'quantile':
        scaledVal = QT.transform(np.array(val).reshape(1, -1))
    return scaledVal[0][0]

# bloodPressure birthdate height weight cholLevel diffWalk heartDisease physHealth physActivity genHealth
@csrf_exempt
def post(request):
    try:
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)

        if 'data' not in data:
            return JsonResponse({'error': "'data' key is required in the request body"}, status=400)

        input = {**data['data']}

        required_keys = ["bloodPressure", "birthdate", "height", "weight", "cholLevel", "diffWalk", "heartDisease", "physHealth", "physActivity", "genHealth"]
        if not(all(key in input for key in required_keys)):
            return JsonResponse({'error': "'Invalid data format"}, status=400)

        input_data = []

        if float(input['bloodPressure']) > 140:
            input['bloodPressure'] = 1
        else:
            input['bloodPressure'] = 0

        if float(input['cholLevel']) > 6:
            input['cholLevel'] = 1
        else:
            input['cholLevel'] = 0

        height = float(input['height'])
        weight = float(input['weight'])
        bmi = weight / height**2

        input['bmi'] = bmi
        input['heartDisease'] = int(input['heartDisease'])
        input['physActivity'] = int(input['physActivity'])
        input['genHealth'] = float(input['genHealth'])
        input['physHealth'] = int(input['physHealth'])
        input['diffWalk'] = int(input['diffWalk'])

        today = datetime.today()
        birthdate = datetime.strptime(input['birthdate'], '%Y-%m-%d')
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        input['age'] = age

        print('input', input)

        input['genHealth'] = scaleVal(input['genHealth'])
        input['bmi'] = scaleVal(input['bmi'])
        input['age'] = scaleVal(input['age'])
        input['physHealth'] = scaleVal(input['physHealth'])

        input['genHealth'] = scaleVal(input['genHealth'], 'quantile')
        input['bmi'] = scaleVal(input['bmi'], 'quantile')
        input['age'] = scaleVal(input['age'], 'quantile')
        input['physHealth'] = scaleVal(input['physHealth'], 'quantile')

        input_data = [input['bloodPressure'], input['cholLevel'], input['bmi'], input['heartDisease'], input['physActivity'], input['genHealth'], input['physHealth'], input['diffWalk'], input['age']]
        input_data = np.array(input_data).reshape(1, 9)

        print('input_data', input_data)

        prediction = model.predict(input_data)

        prediction = prediction.tolist()

        return JsonResponse({'prediction': prediction})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

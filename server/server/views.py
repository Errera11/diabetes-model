import json

import joblib
import numpy as np
from django.http import JsonResponse
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from dateutil import parser

model = tf.keras.models.load_model('model2.keras')
QT = joblib.load('quantile_transformer2.pkl')
ST = joblib.load('standard_transformer2.pkl')


def scaleVal(val):
    parsed = [[val['bloodPressure'], val['cholLevel'], val['bmi'], val['heartDisease'],
              val['physActivity'], val['genHealth'], val['physHealth'], val['diffWalk'],
              val['age']]]

    scaledVal = QT.transform(parsed)
    # print('scaledVal', scaledVal)
    scaledVal = ST.transform(scaledVal)

    return scaledVal[0]


def map_age(age):
    try:
        conditions = [x <= age <= x + 5 for x in range(18, 81, 5)]

        return conditions.index(True) + 1
    except:
        if age > 81:
            return 13

        return JsonResponse({'error': 'Age must be in range 18-...'}, status=400)


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

        required_keys = ["bloodPressure", "birthdate", "height", "weight", "cholLevel", "diffWalk", "heartDisease",
                         "physHealth", "physActivity", "genHealth"]
        if not (all(key in input for key in required_keys)):
            return JsonResponse({'error': "'Invalid data format"}, status=400)

        input_data = {}

        if float(input['bloodPressure']) > 140:
            input_data['bloodPressure'] = 1
        else:
            input_data['bloodPressure'] = 0
        if float(input['cholLevel']) > 6:
            input_data['cholLevel'] = 1
        else:
            input_data['cholLevel'] = 0
        height = float(input['height'])
        weight = float(input['weight'])
        bmi = weight / height ** 2

        input_data['bmi'] = bmi
        input_data['heartDisease'] = int(input['heartDisease'])
        input_data['physActivity'] = int(input['physActivity'])
        input_data['genHealth'] = float(input['genHealth'])
        input_data['physHealth'] = int(input['physHealth'])
        input_data['diffWalk'] = int(input['diffWalk'])

        today = datetime.today()
        birthdate = parser.parse(input['birthdate'])

        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))

        input_data['age'] = map_age(age)

        input_data = scaleVal(input_data)

        input_data = np.array(input_data).reshape(1, 9)

        print('input_data', input_data)
        prediction = model.predict(input_data)

        prediction = prediction.tolist()
        print('prediction', prediction)

        return JsonResponse({'prediction': prediction})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

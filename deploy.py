import pickle
import numpy as np
import gzip
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('house_price.pkl', 'rb'))
#with gzip.open('house_price.sav',"rb") as ifp:
#    print(pickle.load(ifp))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    '''For rendering results on HTML GUI'''

    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
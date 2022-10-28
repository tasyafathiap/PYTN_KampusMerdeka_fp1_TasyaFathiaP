import flask
from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('model/model_reg.pkl', 'rb'))

app = flask.Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(flask.render_template('main2.html'))

@app.route('/linearregression', methods=['POST'])
def linearregression():
    '''
    For rendering results on HTML GUI
    '''
    features = [y for y in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('main2.html', prediction_text='Prediksi Tarif Linear Regression yaitu : $ {}'.format(output))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)
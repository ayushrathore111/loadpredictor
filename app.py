import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

app = Flask(__name__)
lr_loaded = pickle.load(open('./static/lr.pkl','rb'))
xg_loaded = pickle.load(open('./static/xg.pkl','rb'))
etr_loaded = pickle.load(open('./static/etr.pkl','rb'))
gbr_loaded = pickle.load(open('./static/gbr.pkl','rb'))
br_loaded = pickle.load(open('./static/br.pkl','rb'))
rf_loaded = pickle.load(open('./static/rf.pkl','rb'))
import pandas as pd 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    mo= int_features[0]
    prediction=[]
    output=0
    if mo==1:
        int_features= int_features[1:]
        final_features = np.array(int_features)
        pred_lr = lr_loaded.predict([final_features])
        pred_xg = xg_loaded.predict(final_features.reshape(1,-1))
        prediction= (pred_lr+pred_xg)/2

    elif mo==2:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        pred_lr = lr_loaded.predict(final_features)
        pred_rf = rf_loaded.predict(final_features)
        prediction= (pred_lr+pred_rf)/2
    elif mo==3:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        pred_lr = lr_loaded.predict(final_features)
        pred_gbr = gbr_loaded.predict(final_features)
        prediction= (pred_lr+pred_gbr)/2
    elif mo==4:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        pred_br = br_loaded.predict(final_features)
        pred_gbr = gbr_loaded.predict(final_features)
        prediction= (pred_br+pred_gbr)/2
    elif mo==5:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        pred_br = br_loaded.predict(final_features)
        pred_etr = etr_loaded.predict(final_features)
        prediction= (pred_br+pred_etr)/2
    else:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        pred_lr = lr_loaded.predict(final_features)
        pred_br = br_loaded.predict(final_features)
        prediction= (pred_lr+pred_br)/2
        
# dels":["lr&xg","lr&rf","lr&gbr","br&gbr","br&etr","lr&br"],

    output = round(prediction[0], 2)
    print(output)

    return render_template('index.html', prediction_text='Load carrying capacity of sample should be {}kN'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
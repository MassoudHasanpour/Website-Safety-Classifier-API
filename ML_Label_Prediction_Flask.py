# -*- coding: utf-8 -*-
"""
Created on Sun May 16 05:00:11 2021

@author: bp
"""

from flask import Flask, request
import urllib
import requests
import pandas as pd
import numpy as np
import copy
import Preprocessing_Func
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import pickle
import json

app = Flask(__name__)

with open('TrainedModel_and_Inputs.pkl', 'rb') as f:
   [Dom, headers, Unq_Flag, Unq_Title,  most_occur_Titles, Unq_Languages, Unq_Topics, Unq_ML, clf] = pickle.load(f)  

@app.route('/g')
def hello_world():
    """
        this is just for testing server.
        send request to 127.0.0.1:5050
        you must see 'hello world'
    """
    return 'Hello World!'
    


@app.route('/L', methods=['POST'])
def LablePred_Fun():
    print('1')
    data = request.get_json(force=True)
    
    #Data = request.json
    print('2')
    ret = json.loads(data)
    print('2.5')
    df = pd.DataFrame(ret.values(), index=ret.keys())
    print('2.5')
    Data = df.transpose()
    # Data = flask.request.args.get(data)
    # Data = pd.io.json.json_normalize(data)
    
#    urllib.request.urlretrieve(url, "local-filename.jpg")

    obj_Data = Data.select_dtypes(include=['object']).copy()
    Obj_headers = obj_Data.columns
    
    
    selected_columns = Data[["alltime", "disavowed", "removed", "accepted",
               "ExtBackLinks", "RefDomains", "RefIPs", "RefSubNets","RefDomainsEDU",
               "ExtBackLinksEDU", "RefDomainsGOV", "ExtBackLinksGOV", "RefDomainsEDU_Exact", 
               "LastCrawlResult","TrustFlow"]]
    
    Numeric_Headers = ["alltime", "disavowed", "removed", "accepted",
               "ExtBackLinks", "RefDomains", "RefIPs", "RefSubNets","RefDomainsEDU",
               "ExtBackLinksEDU", "RefDomainsGOV", "ExtBackLinksGOV", "RefDomainsEDU_Exact", 
               "LastCrawlResult","TrustFlow"]
    
    Numerics_Data2 = selected_columns.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    Numerics_Data2 = Numerics_Data2.fillna(0)
    Numerics_Data = Numerics_Data2.select_dtypes(include=numerics)
    print('3')
    
    
    if not len(Numerics_Data.columns) == 15:
        Numerics_Data = np.zeros((len(Numerics_Data2), 15), dtype = None, order = 'C')
        for i in range(len(Numeric_Headers)):
            if pd.to_numeric(Numerics_Data2[Numeric_Headers[i]], errors='coerce').notnull().all():
                for j in range(len(Numerics_Data2[Numeric_Headers[i]])):
                    Numerics_Data[j,i] = float(Numerics_Data2[Numeric_Headers[i]][j])
            else:
                for j in range(len(Numerics_Data2[Numeric_Headers[i]])):
                    if not type(Numerics_Data2[Numeric_Headers[i]][j]) == np.int64 and not Numerics_Data2[Numeric_Headers[i]][j].isnumeric():
                        Numerics_Data[j,i] = 0
                    else:
                        Numerics_Data[j,i] = float(Numerics_Data2[Numeric_Headers[i]][j])
    else:
        Numerics_Data = Numerics_Data.to_numpy()       
    print('5')
    
    # In the following lines the data is preprocessed and the desired features are extracted from this
    OneHot_host = Preprocessing_Func.OneHotEncoder_Un(Data["host"], Dom)
    
    OneHot_Flag = Preprocessing_Func.OneHotEncoder_Flag(Data["RedirectFlag"])
    A = np.concatenate((OneHot_Flag, OneHot_host),axis=1)
    print('55')
    OneHot_RedirectTo = Preprocessing_Func.OneHotEncoder_Un(Data["RedirectTo"], Dom)
    print('56')
    A = np.concatenate((A, OneHot_RedirectTo),axis=1)
    
    OneHot_Titles = Preprocessing_Func.OneHotEncoder_Un(Data["Title"], most_occur_Titles)
    A = np.concatenate((A, OneHot_Titles),axis=1)
    
    OneHot_Topics = Preprocessing_Func.OneHotEncoder_Un(Data["TopicalTrustFlow_Topic_0"], Unq_Topics)
    A = np.concatenate((A, OneHot_Topics),axis=1)
    
    OneHot_Language = Preprocessing_Func.OneHotEncoder_Lang(Data["Language"], Data["LanguageConfidence"], Data["LanguagePageRatios"], Unq_Languages)
    Input = np.concatenate((A, OneHot_Language),axis=1)
   
    # Numerics_Data = Numerics_Data.to_numpy()
    Input = np.concatenate((Numerics_Data, Input),axis=1)
    
    y_pred = clf.predict(Input)
    Predicted_ML_Classes = [] # np.zeros((len(y_pred), 1), dtype = None, order = 'C')
    
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            Predicted_ML_Classes.append("bad")
        elif y_pred[i] == 2:
            Predicted_ML_Classes.append("low")
        elif y_pred[i] == 3:
            Predicted_ML_Classes.append("neutral")
        elif y_pred[i] == 4:
            Predicted_ML_Classes.append("suspect")   
        elif y_pred[i] == 5:
            Predicted_ML_Classes.append("good")
  #  print(model_preds)
    print('4')
    Result = json.dumps(Predicted_ML_Classes) # Predicted_ML_Classes.to_json(orient="split")
    print('6')
    return Result
    # return Results
    print('5')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050)

# end = timer()
# print(timedelta(seconds=end-start))

# np.savetxt('model_preds_S.csv', model_preds, delimiter=';')

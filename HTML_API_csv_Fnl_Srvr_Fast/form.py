# importing Flask and other modules
from flask import Flask, request, render_template, jsonify, json
import urllib
import requests
import pandas as pd
import numpy as np
# import copy
import Preprocessing_Func_Fast
import pickle
# import json
# from prettytable import PrettyTable
# import codecs
# import io
# import csv
from keras.models import model_from_json
import keras
import tensorflow as tf
graph = tf.get_default_graph()
  
# Flask constructor
app = Flask(__name__)   

# loaded_model = keras.models.load_model("my_model.h5")

# load json and create model
json_file = open('model_11.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_11.h5")
print("Loaded model from disk")


with open('Inputs_New_Fnl_Fst.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [Dom, Headers, Unq_Languages, 
                 Features_Title, Features_Topics, Features_Topics_v2, 
                 Dict_Dom, Dict_Features_Title, Dict_Features_Topics,
                  Dict_Unq_Languages, scaler] = pickle.load(f)  

Headers_Test = ["Item",	"ResultCode",	"Status",	"ExtBackLinks",	
          "RefDomains",	"ItemType",	"IndexedURLs",	"RefIPs",	
          "RefSubNets",	"RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV",	"LastCrawlDate",	
          "LastCrawlResult",	"Title",	"RedirectTo",	"Language",	
          "LanguageDesc",	"LanguageConfidence",	"RootDomainIPAddress",	
          "CitationFlow",	"TrustFlow",	"TopicalTrustFlow_Topic_0",	
          "TopicalTrustFlow_Value_0",	"Appearances",	"Disavowed",	
          "Accepted"]
# A decorator used to tell the application
# which URL is associated function


@app.route('/LabelPredCSV')
def LablePred_Fun():
    return render_template('InputCSV.html')

@app.route('/LabelPredCSV_Results', methods =["GET", "POST"])
def LablePred_FunR():
    dicts = {}
    Numerics_Data2_Test = []
    if request.method == "POST":
       #f = request.files['file']
       print(1)
       Data_Test = pd.read_csv(request.files.get('file'), encoding= 'unicode_escape',header=None, names=Headers_Test, na_values="?" ,low_memory=False) 
       print(2)
       selected_columns = Data_Test[["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
          "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
          "RefIPs",	"CitationFlow"]]

       Numeric_Headers = ["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
                  "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
                  "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
                  "RefIPs",	"CitationFlow"]
                
       Numerics_Data2_Test = selected_columns.copy()
       numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
       Numerics_Data2_Test = Numerics_Data2_Test.fillna(0)
       Numerics_Data_Test = Numerics_Data2_Test.select_dtypes(include=numerics)
       print('3')
         
         
       if not len(Numerics_Data_Test.columns) == 13:
           Numerics_Data_Test = np.zeros((len(Numerics_Data2_Test), 13), dtype = None, order = 'C')
           for i in range(len(Numeric_Headers)):
               if pd.to_numeric(Numerics_Data2_Test[Numeric_Headers[i]], errors='coerce').notnull().all():
                   for j in range(len(Numerics_Data2_Test[Numeric_Headers[i]])):
                       Numerics_Data_Test[j,i] = float(Numerics_Data2_Test[Numeric_Headers[i]][j])
               else:
                   for j in range(len(Numerics_Data2_Test[Numeric_Headers[i]])):
                       if not type(Numerics_Data2_Test[Numeric_Headers[i]][j]) == np.int64 and not type(Numerics_Data2_Test[Numeric_Headers[i]][j]) == int and not Numerics_Data2_Test[Numeric_Headers[i]][j].isnumeric():
                           Numerics_Data_Test[j,i] = 0
                       else:
                           Numerics_Data_Test[j,i] = float(Numerics_Data2_Test[Numeric_Headers[i]][j])
       else:
            Numerics_Data_Test = Numerics_Data_Test.to_numpy()      
                
       Numerics_Data_Test_Norm = scaler.transform(Numerics_Data_Test)
       print('4') 
        
       OneHot_Item_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["Item"], Dict_Dom)
       print('44') 
       OneHot_Titless_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["Title"], Dict_Features_Title)
       print('444') 
       A = np.concatenate((OneHot_Item_Test, OneHot_Titless_Test),axis=1)
       OneHot_Topicss_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["TopicalTrustFlow_Topic_0"], Dict_Features_Topics)
       A = np.concatenate((A, OneHot_Topicss_Test),axis=1)
       print('4444') 
       OneHot_Language_Test = Preprocessing_Func_Fast.OneHotEncoder_Langg(Data_Test["Language"], Data_Test["LanguageConfidence"], Unq_Languages)
       Input_Test = np.concatenate((A, OneHot_Language_Test),axis=1)
       print('5444') 
       Input_Test = np.concatenate((Numerics_Data_Test_Norm, Input_Test),axis=1)
        
       with graph.as_default(): 
           y_pred = loaded_model.predict(Input_Test)
       y_pred_class = np.argmax(y_pred, axis=1)
        
       ML_Classes_Test_2_1 = []
       print('6')      
            
       for i in range(len(y_pred_class)):
           if y_pred_class[i] == 0: # "bad":
               ML_Classes_Test_2_1.append("bad")
           elif y_pred_class[i] == 1: # "low":
               ML_Classes_Test_2_1.append("low")
           elif y_pred_class[i] == 2: # "suspect":
               ML_Classes_Test_2_1.append("suspect")  
           elif y_pred_class[i] == 3: # "good" or Data['ML'][i] == "Good":
               ML_Classes_Test_2_1.append("good")     
      #  print(model_preds)
       print('7')
   #    Result = json.dumps(Predicted_ML_Classes) # Predicted_ML_Classes.to_json(orient="split")
   #    headers = ['Index', 'Labels Predicted By ML']
       # Table = [] # PrettyTable(['Index', 'Labels Predicted By ML'])

       # for i in range(len(Predicted_ML_Classes)):
       #    Table.append([str(i), Predicted_ML_Classes[i]])
       # print('6')
       
       dicts = {}
       dicts['Index'] = 'Labels Predicted By ML'    
       for i in range(len(ML_Classes_Test_2_1)):
           dicts[str(i+1)] = ML_Classes_Test_2_1[i]
       
        # jsonify(Table) # Result
        # return Results
       print('8')
      # return dicts
       return render_template("ResultCSV.html", dicts = dicts)
  # host="0.0.0.0", port=8050
@app.route('/LabelPredJson', methods =["GET", "POST"])
def LablePred_FunnR():
    dicts = {}
    Numerics_Data2_Test = []
    if request.method == "POST":
       #f = request.files['file']
       print(1)
       data = json.loads(request.data)
       Data_Test = pd.DataFrame.from_dict(data, orient="index")
       # Data_Test = pd.read_csv(request.files.get('file'), encoding= 'unicode_escape',header=None, names=Headers_Test, na_values="?" ,low_memory=False) 
       print(2)
       selected_columns = Data_Test[["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
          "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
          "RefIPs",	"CitationFlow"]]

       Numeric_Headers = ["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
                  "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
                  "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
                  "RefIPs",	"CitationFlow"]
                
       Numerics_Data2_Test = selected_columns.copy()
       numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
       Numerics_Data2_Test = Numerics_Data2_Test.fillna(0)
       Numerics_Data_Test = Numerics_Data2_Test.select_dtypes(include=numerics)
       print('3')
         
         
       if not len(Numerics_Data_Test.columns) == 13:
           Numerics_Data_Test = np.zeros((len(Numerics_Data2_Test), 13), dtype = None, order = 'C')
           for i in range(len(Numeric_Headers)):
               if pd.to_numeric(Numerics_Data2_Test[Numeric_Headers[i]], errors='coerce').notnull().all():
                   for j in range(len(Numerics_Data2_Test[Numeric_Headers[i]])):
                       Numerics_Data_Test[j,i] = float(Numerics_Data2_Test[Numeric_Headers[i]][j])
               else:
                   for j in range(len(Numerics_Data2_Test[Numeric_Headers[i]])):
                       if not type(Numerics_Data2_Test[Numeric_Headers[i]][j]) == np.int64 and not type(Numerics_Data2_Test[Numeric_Headers[i]][j]) == int and not Numerics_Data2_Test[Numeric_Headers[i]][j].isnumeric():
                           Numerics_Data_Test[j,i] = 0
                       else:
                           Numerics_Data_Test[j,i] = float(Numerics_Data2_Test[Numeric_Headers[i]][j])
       else:
            Numerics_Data_Test = Numerics_Data_Test.to_numpy()      
                
       Numerics_Data_Test_Norm = scaler.transform(Numerics_Data_Test)
       print('4') 
        
       OneHot_Item_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["Item"], Dict_Dom)
       print('44') 
       OneHot_Titless_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["Title"], Dict_Features_Title)
       print('444') 
       A = np.concatenate((OneHot_Item_Test, OneHot_Titless_Test),axis=1)
       OneHot_Topicss_Test = Preprocessing_Func_Fast.OneHotEncoder_Un(Data_Test["TopicalTrustFlow_Topic_0"], Dict_Features_Topics)
       A = np.concatenate((A, OneHot_Topicss_Test),axis=1)
       print('4444') 
       OneHot_Language_Test = Preprocessing_Func_Fast.OneHotEncoder_Langg(Data_Test["Language"], Data_Test["LanguageConfidence"], Unq_Languages)
       Input_Test = np.concatenate((A, OneHot_Language_Test),axis=1)
       print('5444') 
       Input_Test = np.concatenate((Numerics_Data_Test_Norm, Input_Test),axis=1)
        
       with graph.as_default(): 
           y_pred = loaded_model.predict(Input_Test)
       y_pred_class = np.argmax(y_pred, axis=1)
        
       ML_Classes_Test_2_1 = []
       print('6')      
            
       for i in range(len(y_pred_class)):
           if y_pred_class[i] == 0: # "bad":
               ML_Classes_Test_2_1.append("bad")
           elif y_pred_class[i] == 1: # "low":
               ML_Classes_Test_2_1.append("low")
           elif y_pred_class[i] == 2: # "suspect":
               ML_Classes_Test_2_1.append("suspect")  
           elif y_pred_class[i] == 3: # "good" or Data['ML'][i] == "Good":
               ML_Classes_Test_2_1.append("good")     
      #  print(model_preds)
       print('7')
   #    Result = json.dumps(Predicted_ML_Classes) # Predicted_ML_Classes.to_json(orient="split")
   #    headers = ['Index', 'Labels Predicted By ML']
       # Table = [] # PrettyTable(['Index', 'Labels Predicted By ML'])

       # for i in range(len(Predicted_ML_Classes)):
       #    Table.append([str(i), Predicted_ML_Classes[i]])
       # print('6')
       
       dicts = {}
       dicts['Index'] = 'Labels Predicted By ML'    
       for i in range(len(ML_Classes_Test_2_1)):
           dicts[str(i+1)] = ML_Classes_Test_2_1[i]
       
        # jsonify(Table) # Result
        # return Results
       print('8')
      # return dicts
       Jsons = jsonify(dicts)
       print('9')
       return Jsons  # render_template("ResultCSV.html", dicts = dicts)
  # host="0.0.0.0", port=8050
if __name__=='__main__':
   app.run(host="0.0.0.0", port=8050)
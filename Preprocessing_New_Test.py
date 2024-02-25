# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 23:00:46 2021

@author: hasan
"""



import pandas as pd
import numpy as np
import copy
import Preprocessing_Func

import pickle


# with open('Inputs_New.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
#     [Data, Dom, Headers, Unq_Title,  Unique_Title_v2, Unq_Languages, 
#                  Features_Title, Features_Topics, OneHot_Item, OneHot_Titles, 
#                  OneHot_Topics, Input, Numerics_Data, ML_Classes] = pickle.load(f)   

with open('Inputs_New_2.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [Data, Dom, Headers, Unq_Title,  Unique_Title_v2, Unq_Languages, 
                 Features_Title, Features_Topics, OneHot_Item, OneHot_Titles, 
                 OneHot_Topics, Input, Numerics_Data, ML_Classes, scaler] = pickle.load(f)       

with open('Inputs_New_Test.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [Data, Dom, Headers_Test, Unq_Title,  Unique_Title_v2,
     Unq_Languages, 
                 Features_Title, Features_Topics, OneHot_Item_Test, 
                 OneHot_Topicss_Test, OneHot_Titless_Test, OneHot_Language_Test,
                 Input_Test, Numerics_Data_Test, Numerics_Data2_Test, scaler] = pickle.load(f) 

Headers= ["Item",	"ML",	"ResultCode",	"Status",	"ExtBackLinks",	
          "RefDomains",	"ItemType",	"IndexedURLs",	"RefIPs",	
          "RefSubNets",	"RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV",	"LastCrawlDate",	
          "LastCrawlResult",	"Title",	"RedirectTo",	"Language",	
          "LanguageDesc",	"LanguageConfidence",	"RootDomainIPAddress",	
          "CitationFlow",	"TrustFlow",	"TopicalTrustFlow_Topic_0",	
          "TopicalTrustFlow_Value_0",	"Appearances",	"Disavowed",	
          "Accepted"]

# Headers_Test = ['Item',	'Appearances',	'Disavowed', 'Accepted', 'ResultCode',
#                 'Status','ExtBackLinks','RefDomains', 'ItemType', 'IndexedURLs',
#                 'RefIPs', 'RefSubNets',	'RefDomainsEDU', 'ExtBackLinksEDU',	
#                 'RefDomainsGOV', 'ExtBackLinksGOV',	'LastCrawlDate', 
#                 'LastCrawlResult', 'Title', 'RedirectTo', 'Language', 
#                 'LanguageDesc', 'LanguageConfidence', 'RootDomainIPAddress', 
#                 'CitationFlow', 'TrustFlow', 'TopicalTrustFlow_Topic_0', 
#                 'TopicalTrustFlow_Value_0']

Headers_Test = ["Item",	"ResultCode",	"Status",	"ExtBackLinks",	
          "RefDomains",	"ItemType",	"IndexedURLs",	"RefIPs",	
          "RefSubNets",	"RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV",	"LastCrawlDate",	
          "LastCrawlResult",	"Title",	"RedirectTo",	"Language",	
          "LanguageDesc",	"LanguageConfidence",	"RootDomainIPAddress",	
          "CitationFlow",	"TrustFlow",	"TopicalTrustFlow_Topic_0",	
          "TopicalTrustFlow_Value_0",	"Appearances",	"Disavowed",	
          "Accepted"]

Data_Test = pd.read_csv("Shahin_2022_1-NewTest.csv", encoding= 'unicode_escape',header=None, names=Headers_Test, na_values="?" ,low_memory=False)
Data = pd.read_csv("NewMLmodel.csv", encoding= 'unicode_escape',header=None, names=Headers, na_values="?" ,low_memory=False)
# Data['ML'].value_counts()
# Include object
obj_Data = Data.select_dtypes(include=['object']).copy()
Obj_headers = obj_Data.columns


# Include numerics
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# Numerics_Data = Data.select_dtypes(include=numerics)
# Numerics_Data = Numerics_Data.fillna(0)

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

# headers = ["host","alltime", "disavowed", "removed", "accepted", "ml",
#            "ExtBackLinks", "RefDomains", "RefIPs", "RefSubNets","RefDomainsEDU",
#            "ExtBackLinksEDU", "RefDomainsGOV", "ExtBackLinksGOV", "RefDomainsEDU_Exact", 
#            "LastCrawlResult", "RedirectFlag", "Title", "RedirectTo", 
#            "Language", "LanguageDesc", "LanguageConfidence", "LanguagePageRatios",
#            "CitationFlow", "TrustFlow", "TopicalTrustFlow_Topic_0", 
#            "TopicalTrustFlow_Value_0"]

Dom2 = [".com",".com.tr", ".com.au", ".bt.com",".org",".org.uk", ".net", ".html", ".htm", ".gov",
           ".edu", ".edu.au", ".fm", ".ac.uk", ".cn",".de",
           ".int", ".europa.eu", ".org.uk", ".gov.au", "nih.gov", ".info", ".press",
           ".gov.uk", ".it", ".co.jp", ".org.uk", ".fr", ".biz", ".us", ".in",
           ".cz", ".ac.nz", ".ru", ".edu.hk", ".es", ".ru", ".xyz", ".news",
           ".berkeley.edu", ".nl", ".se", ".opx.pl", ".co", ".co.uk", ".sc",
           ".blog", ".edu.sg", ".ie", ".cybo.com", ".in.th", ".ir", "jp",".io",
           ".eu", ".tf","ee",".tv", ".mn", ".nu", ".lu", "media", ".uk", ".video",
           ".club", ".online", ".onl", ".ooo", ".earth", ".asia", ".earth", ".rs",
           ".nz", ".me", ".vacations", ".rentals", ".events",".casino", ".ws", ".be",
           ".su", ".store", ".marketing", ".market", ".exchange", ".fund", ".ca",
           ".gl", ".tips", ".yt", ".sk", ".photos", ".kz", ".ml", ".my", "fi", ".ch"
           ".university", ".pl", ".pt", ".ua", ".id", ".dk", ".tk", ".is", ".cc", ".mx",
           ".website", ".tech", ".site", ".space", ".hu", ".top", ".hr", ".rocks",
           ".at", ".ki", ".af", ".cf", ".ge", ".ps", ".stream", ".shop", ".hk", ".kr",
           ".ch", ".ly", ".sg", ".no", ".lt", ".do", ".ro", ".pro", ".li", ".tw", ".si",
           ".agency", ".world", ".to", ".tr", ".pw", ".mobi", ".tw", ".bg", "work",
           ".reviews", ".fashion", ".la", ".gs", ".bz", "lk", ".cl", "show", ".poker",
           ".vg", ".wf", ".credit", ".pe", ".pm", "cl", ".cx", ".games", ".lv", ".jewelry",
           ".ac", ".gr", ".im", ".ng", ".name", ".am", ".lc", ".bingo", ".gy", ".diamonds",
           ".equipment", "job", ".loans", "advertis", "globe", "theworld", ".tl", 
           ".ga", ".md", ".so", ".ae", ".pk", "business", "earth", ".wiki", ".ai"]

# for Domain in Data["RedirectTo"]:
#     k = 0
#     for extension in Dom:
#         if Domain == Domain and extension in Domain:
#             k += 1
#     if k == 0 and Domain == Domain:
#         print(Domain)

# Data = pd.read_csv("ml_2.csv", encoding= 'unicode_escape',header=None, names=headers, na_values="?" ,low_memory=False)

# Include object
obj_Data_Test = Data_Test.select_dtypes(include=['object']).copy()
obj_Data_Test.head()


# Unq_Flag = ["True", "False"]# Preprocessing_Func.Unique_Str(Data["RedirectFlag"])
# Unq_Title = Preprocessing_Func.Unique_Str(Data["Title"])
# Unique_Title = Preprocessing_Func.Unique_LvL2(Data['Title'], " ")

# j = 0
# Unique_Title_v2 = []
# for i in range(len(Unique_Title)):
#     if len(Unique_Title[i])>3:
#         Unique_Title_v2.append(Unique_Title[i])


# OneHot_Titles = Preprocessing_Func.OneHotEncoder_Un(Data["Title"], Unique_Title_v2)


# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt

# X_train, X_test, y_train, y_test = train_test_split(OneHot_Titles, ML_Classes, test_size=0.2, random_state=42, shuffle=True)

# model = RandomForestClassifier()
# model.fit(X_train,y_train)
# Ftur_Imp_Tilte = model.feature_importances_
# print(model.feature_importances_)

# Features_Title = []
# for i in range(len(Ftur_Imp_Tilte)):
#     if Ftur_Imp_Tilte[i] > 0.000045:
#         Features_Title.append(Unique_Title_v2[i])

# Unq_Topics = Preprocessing_Func.Unique_LvL2(Data["TopicalTrustFlow_Topic_0"], "/")
# OneHot_Topics = Preprocessing_Func.OneHotEncoder_Un(Data["TopicalTrustFlow_Topic_0"], Unq_Topics)
# X_train, X_test, y_train, y_test = train_test_split(OneHot_Topics, Data["ML"], test_size=0.4, random_state=42, shuffle=True)

# model = ExtraTreesClassifier()
# model.fit(X_train,y_train)
# Ftur_Imp_Topics = model.feature_importances_
# print(model.feature_importances_)

# Features_Topics = []
# for i in range(len(Ftur_Imp_Topics)):
#     if Ftur_Imp_Topics[i] > 0.00004:
#         Features_Topics.append(Unq_Topics[i])


# most_occur_Titles = Preprocessing_Func.most_Occured(Unq_Title)
        
Unq_Languages = Preprocessing_Func.Unique_LvL2(Data["Language"], ",")
# Unq_ML = Preprocessing_Func.Unique_Str(Data["ml"])

OneHot_Item_Test = Preprocessing_Func.OneHotEncoder_Un(Data_Test["Item"], Dom)
OneHot_Titless_Test = Preprocessing_Func.OneHotEncoder_Un(Data_Test["Title"], Features_Title)
# OneHot_Flag = Preprocessing_Func.OneHotEncoder_Flag(Data["RedirectFlag"])
A = np.concatenate((OneHot_Item_Test, OneHot_Titless_Test),axis=1)
# OneHot_RedirectTo = Preprocessing_Func.OneHotEncoder_Un(Data["RedirectTo"], Dom)
# A = np.concatenate((A, OneHot_RedirectTo),axis=1)
# A = np.concatenate((A, OneHot_Titles),axis=1)
OneHot_Topicss_Test = Preprocessing_Func.OneHotEncoder_Un(Data_Test["TopicalTrustFlow_Topic_0"], Features_Topics)
A = np.concatenate((A, OneHot_Topicss_Test),axis=1)
OneHot_Language_Test = Preprocessing_Func.OneHotEncoder_Langg(Data_Test["Language"], Data_Test["LanguageConfidence"], Unq_Languages)
Input_Test = np.concatenate((A, OneHot_Language_Test),axis=1)

Input_Test = np.concatenate((Numerics_Data_Test_Norm, Input_Test),axis=1)

# ML_Classes = Data['ML']

# cleanup_nums = {"ml": {"bad": 1, "low": 2, "neutral": 3, "suspect": 4,
#                                  "good": 5}}

y_pred = modell.predict(Input_Test)
y_pred_class = np.argmax(y_pred, axis=1)

ML_Classes_Test_1 = []
     
    
for i in range(len(y_pred_class)):
    if y_pred_class[i] == 0: # "bad":
        ML_Classes_Test_1.append("bad")
    elif y_pred_class[i] == 1: # "low":
        ML_Classes_Test_1.append("low")
    elif y_pred_class[i] == 2: # "suspect":
        ML_Classes_Test_1.append("suspect")  
    elif y_pred_class[i] == 3: # "good" or Data['ML'][i] == "Good":
        ML_Classes_Test_1.append("good")  

df = pd.DataFrame(ML_Classes_Test_1)
df.to_csv('ML_Classes_Test_modell.csv') 

modell_json = modell.to_json()
with open("modell.json", "w") as json_file:
    json_file.write(modell_json)
# serialize weights to HDF5
modell.save_weights("modell.h5")
print("Saved model to disk")
   
y_pred = model.predict(Input_Test)
y_pred_class = np.argmax(y_pred, axis=1)

ML_Classes_Test_2 = []
     
    
for i in range(len(y_pred_class)):
    if y_pred_class[i] == 0: # "bad":
        ML_Classes_Test_2.append("bad")
    elif y_pred_class[i] == 1: # "low":
        ML_Classes_Test_2.append("low")
    elif y_pred_class[i] == 2: # "suspect":
        ML_Classes_Test_2.append("suspect")  
    elif y_pred_class[i] == 3: # "good" or Data['ML'][i] == "Good":
        ML_Classes_Test_2.append("good")     
      
df = pd.DataFrame(ML_Classes_Test_2)
df.to_csv('ML_Classes_Test_2.csv')         



model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

from keras.models import model_from_json

# load json and create model
json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_1.h5")
print("Loaded model from disk")




y_pred = model_1.predict(Input_Test)
y_pred_class = np.argmax(y_pred, axis=1)

ML_Classes_Test_2_1 = []
     
    
for i in range(len(y_pred_class)):
    if y_pred_class[i] == 0: # "bad":
        ML_Classes_Test_2_1.append("bad")
    elif y_pred_class[i] == 1: # "low":
        ML_Classes_Test_2_1.append("low")
    elif y_pred_class[i] == 2: # "suspect":
        ML_Classes_Test_2_1.append("suspect")  
    elif y_pred_class[i] == 3: # "good" or Data['ML'][i] == "Good":
        ML_Classes_Test_2_1.append("good")     
      
df = pd.DataFrame(ML_Classes_Test_2_1)
df.to_csv('ML_Classes_Test_ll2_model_1.csv')         



model_json = model_1.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_1.save_weights("model_1.h5")
print("Saved model to disk")

model_json = model_11.to_json()
with open("model_11.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_11.save_weights("model_11.h5")
print("Saved model to disk")

from keras.models import model_from_json

# load json and create model
json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_1.h5")
print("Loaded model from disk")
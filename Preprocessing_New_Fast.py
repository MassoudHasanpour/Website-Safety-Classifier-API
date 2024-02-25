# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 22:30:03 2022

@author: hasan
"""

import pandas as pd
import numpy as np
import copy
import Preprocessing_Func_Fast


Headers= ["Item",	"ML",	"ResultCode",	"Status",	"ExtBackLinks",	
          "RefDomains",	"ItemType",	"IndexedURLs",	"RefIPs",	
          "RefSubNets",	"RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV",	"LastCrawlDate",	
          "LastCrawlResult",	"Title",	"RedirectTo",	"Language",	
          "LanguageDesc",	"LanguageConfidence",	"RootDomainIPAddress",	
          "CitationFlow",	"TrustFlow",	"TopicalTrustFlow_Topic_0",	
          "TopicalTrustFlow_Value_0",	"Appearances",	"Disavowed",	
          "Accepted"]

Headers_Test = ['Item',	'Appearances',	'Disavowed', 'Accepted', 'ResultCode',
                'Status','ExtBackLinks','RefDomains', 'ItemType', 'IndexedURLs',
                'RefIPs', 'RefSubNets',	'RefDomainsEDU', 'ExtBackLinksEDU',	
                'RefDomainsGOV', 'ExtBackLinksGOV',	'LastCrawlDate', 
                'LastCrawlResult', 'Title', 'RedirectTo', 'Language', 
                'LanguageDesc', 'LanguageConfidence', 'RootDomainIPAddress', 
                'CitationFlow', 'TrustFlow', 'TopicalTrustFlow_Topic_0', 
                'TopicalTrustFlow_Value_0']

Data_Test = pd.read_csv("NewMLmodel.csv", encoding= 'unicode_escape',header=None, names=Headers_Test, na_values="?" ,low_memory=False)
Data = pd.read_csv("NewMLmodel.csv", encoding= 'unicode_escape',header=None, names=Headers, na_values="?" ,low_memory=False)
Data['ML'].value_counts()
# Include object
obj_Data = Data.select_dtypes(include=['object']).copy()
Obj_headers = obj_Data.columns


# Include numerics
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# Numerics_Data = Data.select_dtypes(include=numerics)
# Numerics_Data = Numerics_Data.fillna(0)

selected_columns = Data[["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
          "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
          "RefIPs",	"CitationFlow"]]

Numeric_Headers = ["Disavowed", "Accepted", "TrustFlow", "ExtBackLinks",	
          "RefDomains", "RefDomainsEDU",	"ExtBackLinksEDU",	
          "RefDomainsGOV",	"ExtBackLinksGOV", "Appearances",	"IndexedURLs",	
          "RefIPs",	"CitationFlow"]
        
Numerics_Data2 = selected_columns.copy()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Numerics_Data2 = Numerics_Data2.fillna(0)
Numerics_Data = Numerics_Data2.select_dtypes(include=numerics)
print('3')
 
 
if not len(Numerics_Data.columns) == 13:
    Numerics_Data = np.zeros((len(Numerics_Data2), 13), dtype = None, order = 'C')
    for i in range(len(Numeric_Headers)):
        if pd.to_numeric(Numerics_Data2[Numeric_Headers[i]], errors='coerce').notnull().all():
            for j in range(len(Numerics_Data2[Numeric_Headers[i]])):
                Numerics_Data[j,i] = float(Numerics_Data2[Numeric_Headers[i]][j])
        else:
            for j in range(len(Numerics_Data2[Numeric_Headers[i]])):
                if not type(Numerics_Data2[Numeric_Headers[i]][j]) == np.int64 and not type(Numerics_Data2[Numeric_Headers[i]][j]) == int and not Numerics_Data2[Numeric_Headers[i]][j].isnumeric():
                    Numerics_Data[j,i] = 0
                else:
                    Numerics_Data[j,i] = float(Numerics_Data2[Numeric_Headers[i]][j])
else:
    Numerics_Data = Numerics_Data.to_numpy()      
        


# headers = ["host","alltime", "disavowed", "removed", "accepted", "ml",
#            "ExtBackLinks", "RefDomains", "RefIPs", "RefSubNets","RefDomainsEDU",
#            "ExtBackLinksEDU", "RefDomainsGOV", "ExtBackLinksGOV", "RefDomainsEDU_Exact", 
#            "LastCrawlResult", "RedirectFlag", "Title", "RedirectTo", 
#            "Language", "LanguageDesc", "LanguageConfidence", "LanguagePageRatios",
#            "CitationFlow", "TrustFlow", "TopicalTrustFlow_Topic_0", 
#            "TopicalTrustFlow_Value_0"]

Dom = [".com",".com.tr", ".com.au", ".bt.com",".org",".org.uk", ".net", ".html", ".htm", ".gov",
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
obj_Data = Data.select_dtypes(include=['object']).copy()
obj_Data.head()


Unq_Flag = ["True", "False"]# Preprocessing_Func.Unique_Str(Data["RedirectFlag"])
# Unq_Title = Preprocessing_Func_Fast.Unique_Str(Data["Title"])
Unique_Title = Preprocessing_Func_Fast.Unique_LvL2(Data['Title'], " ")

# AA = pd.DataFrame(Unique_Title)
# Unique_Topics_v2 = Preprocessing_Func.Unique_LvL2(AA[0], " ")

j = 0
Unique_Title_v2 = []
for i in range(len(Unique_Title)):
    if len(Unique_Title[i])>3:
        Unique_Title_v2.append(Unique_Title[i])


j= 0
Dict_Features_Title_v1 = {}
for x in Unique_Title:
    Dict_Features_Title_v1[x] = j
    j+=1




OneHot_Titles = Preprocessing_Func_Fast.OneHotEncoder_Un(Data["Title"], Dict_Features_Title_v1)

ML_Classes = Data['ML']
# ML_Classes = np.zeros((len(Data['ML']), 1), dtype = None, order = 'C') # .to_list()

# for i in range(len(Data['ML'])):
#     if Data['ML'][i] == "bad":
#         ML_Classes[i,0] = 0
#     elif Data['ML'][i] == "low":
#         ML_Classes[i,0] = 1
#     # elif Data['ml'][i] == "neutral":
#     #     ML_Classes[i,0] = 3
#     elif Data['ML'][i] == "suspect":
#         ML_Classes[i,0] = 2   
#     elif Data['ML'][i] == "good" or Data['ML'][i] == "Good":
#         ML_Classes[i,0] = 3       
    

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
# import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(OneHot_Titles, ML_Classes, test_size=0.2, random_state=42, shuffle=True)

model = RandomForestClassifier()
model.fit(X_train,y_train.values.ravel())
Ftur_Imp_Tilte = model.feature_importances_
print(model.feature_importances_)

Features_Title = []
for i in range(len(Ftur_Imp_Tilte)):
    if Ftur_Imp_Tilte[i] > 0.00005:
        Features_Title.append(Unique_Title[i])

# j= 0
# Dict_Features_Topics_v1={}
# for x in Features_Topics:
#     Dict_Features_Topics_v1[x] = j
#     j+=1


# Unq_Topics = Preprocessing_Func_Fast.Unique_LvL2(Data["TopicalTrustFlow_Topic_0"], "/")
OneHot_Topics = Preprocessing_Func_Fast.OneHotEncoder_Un(Data["TopicalTrustFlow_Topic_0"], Features_Topics_v2)
X_train, X_test, y_train, y_test = train_test_split(OneHot_Topics, Data["ML"], test_size=0.4, random_state=42, shuffle=True)

model = RandomForestClassifier()
model.fit(X_train,y_train)
Ftur_Imp_Topics = model.feature_importances_
print(model.feature_importances_)

Unq_Languages = Preprocessing_Func_Fast.Unique_LvL2_Lang(Data["Language"], ",")


Features_Topics = []
for i in range(len(Ftur_Imp_Topics)):
    if Ftur_Imp_Topics[i] > 0.0001:
        Features_Topics.append(Unique_Topics_v2[i])

j= 0
Dict_Features_Topics={}
for x in Features_Topics:
    Dict_Features_Topics[x] = j
    j+=1

j= 0
Dict_Features_Title = {}
for x in Features_Title:
    Dict_Features_Title[x] = j
    j+=1

j= 0
Dict_Dom={}
for x in Dom:
    Dict_Dom[x] = j
    j+=1

j= 0
Dict_Unq_Languages={}
for x in Unq_Languages:
    Dict_Unq_Languages[x] = j
    j+=1
    
# most_occur_Titles = Preprocessing_Func.most_Occured(Unq_Title)
        
# Unq_ML = Preprocessing_Func.Unique_Str(Data["ml"])

OneHot_Item = Preprocessing_Func_Fast.OneHotEncoder_Un(Data["Item"], Dict_Dom)
OneHot_Titless = Preprocessing_Func_Fast.OneHotEncoder_Un(Data["Title"], Dict_Features_Title)
# OneHot_Flag = Preprocessing_Func.OneHotEncoder_Flag(Data["RedirectFlag"])
A = np.concatenate((OneHot_Item, OneHot_Titless),axis=1)
# OneHot_RedirectTo = Preprocessing_Func.OneHotEncoder_Un(Data["RedirectTo"], Dom)
# A = np.concatenate((A, OneHot_RedirectTo),axis=1)
# A = np.concatenate((A, OneHot_Titles),axis=1)
OneHot_Topicss = Preprocessing_Func_Fast.OneHotEncoder_Un(Data["TopicalTrustFlow_Topic_0"], Dict_Features_Topics)
A = np.concatenate((A, OneHot_Topicss),axis=1)
OneHot_Language = Preprocessing_Func_Fast.OneHotEncoder_Langg(Data["Language"], Data["LanguageConfidence"], Unq_Languages)
Input = np.concatenate((A, OneHot_Language),axis=1)

Input = np.concatenate((Numerics_Data, Input),axis=1)

# ML_Classes = Data['ML']

# cleanup_nums = {"ml": {"bad": 1, "low": 2, "neutral": 3, "suspect": 4,
#                                  "good": 5}}



import pickle

# Saving the objects:
with open('Inputs_New.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Data, Dom, Headers, Unq_Title,  Unique_Title_v2, Unq_Languages, 
                 Features_Title, Features_Topics, OneHot_Item, OneHot_Titles, 
                 OneHot_Topics, OneHot_Titless, OneHot_Language,
                 OneHot_Topicss, Input, Numerics_Data, ML_Classes], f)
    
with open('Inputs_New.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [Data, Dom, Headers, Unq_Title,  Unique_Title_v2, Unq_Languages, 
                 Features_Title, Features_Topics, OneHot_Item, OneHot_Titles, 
                 OneHot_Topics, OneHot_Titless, OneHot_Language,
                 OneHot_Topicss, Input, Numerics_Data, ML_Classes] = pickle.load(f)   
   
   
with open('Inputs_New_2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Data, Dom, Headers, Unq_Title,  Unique_Title_v2,
     Unique_Topics_v2, 
    Unq_Languages, 
                 Features_Title, Features_Topics, Features_Topics_v2, 
                 OneHot_Item, OneHot_Titles, 
                 OneHot_Topics, OneHot_Titless, OneHot_Language,
                 OneHot_Topicss, Input, Numerics_Data, Dict_Dom, 
                 Dict_Features_Title, Dict_Features_Topics, Dict_Unq_Languages,
                 ML_Classes, scaler], f)    
    
with open('Inputs_New_2.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    [Data, Dom, Headers, Unq_Title,  Unique_Title_v2,
     Unique_Topics_v2, Unq_Languages, 
                 Features_Title, Features_Topics, Features_Topics_v2, 
                 OneHot_Item, OneHot_Titles, 
                 OneHot_Topics, OneHot_Titless, OneHot_Language,
                 OneHot_Topicss, Input, Numerics_Data, Dict_Dom, 
                 Dict_Features_Title, Dict_Features_Topics, Dict_Unq_Languages,
                 ML_Classes, scaler] = pickle.load(f)       
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:15:00 2021

@author: hasan
"""

import requests
import pandas as pd
import json
import pickle
from prettytable import PrettyTable

headers = ["host","alltime", "disavowed", "removed", "accepted",
           "ExtBackLinks", "RefDomains", "RefIPs", 
           
           "RefSubNets","RefDomainsEDU",
           "ExtBackLinksEDU", "RefDomainsGOV", "ExtBackLinksGOV",
           "RefDomainsEDU_Exact", 
           "LastCrawlResult", "RedirectFlag", 
           "Title", "RedirectTo", 
           "Language", "LanguageDesc", "LanguageConfidence", "LanguagePageRatios",
           "CitationFlow", "TrustFlow", "TopicalTrustFlow_Topic_0", 
           "TopicalTrustFlow_Value_0"]    
Data = pd.read_csv(r".\Test.csv", encoding= 'unicode_escape',header=None, names=headers, na_values="?" ,low_memory=False)
result = Data.to_json(orient="columns")

Address = "http://195.191.164.121:8050"
# Address = "http://192.168.1.101:1111"
url = Address + '/Label_API'
rslt = requests.post(url, json=result).content

Result = json.loads(rslt)

from prettytable import PrettyTable
Table = []
Table = PrettyTable(['Index', 'Label Predicted By ML'])
# 
for i in range(len(Result)):
    Table.add_row([str(i), Result[i]])
   
    
print(Table)

# print(Result)
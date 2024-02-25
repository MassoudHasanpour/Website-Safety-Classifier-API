# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:10:26 2021

@author: hasan
"""
import numpy as np
import re
from collections import Counter

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def Unique_Str(Array):
    output = []
    for x in Array:
        if x not in output and x == x:
            output.append(x)
    return output

def Unique_LvL2(Array, ch):
    Array_new = Array[Array.notnull()]
    output = []
    for x in Array_new:
        Ind = find(x ,ch)
        if len(Ind)>0:
            obj = x.split(ch)
            for y in obj:
                if y not in output and not y.isnumeric():
                    output.append(y)
        else:
            if x not in output and x == x:
                output.append(x)
    return output

def OneHotEncoder_Un(Array, Unique_Els):
    Output = np.zeros((len(Array), len(Unique_Els)), dtype = None, order = 'C')
    for i in range(len(Array)):
        for j in range(len(Unique_Els)):
            if Array[i] == Array[i] and not Array[i] == None and Unique_Els[j] in Array[i]:
                Output[i,j] = 1
    return Output

def OneHotEncoder_Flag(Array):
    Output = np.zeros((len(Array), 2), dtype = None, order = 'C')
    for i in range(len(Array)):
        
        if Array[i] == Array[i] and Array[i]:
            Output[i,0] = 1
        elif Array[i] == Array[i] and not Array[i]:
            Output[i,1] = 1
    return Output


def OneHotEncoder_Lang(Array1, Array2, Array3, Unique_Els):
    Output = np.zeros((len(Array1), 2*len(Unique_Els)), dtype = None, order = 'C')
    for i in range(len(Array1)):
        X1 = Array1[i]
        X2 = Array2[i]
        X3 = Array3[i]
        if X1 == X1 and X2 == X2 and X3 == X3 and not X1 == None and not X2 == None and not X3 == None:
            for j in range(len(Unique_Els)):
                if  "," in X1 and Unique_Els[j] in X1 and len(Unique_Els[j]) <3:
                    X11 = X1.split(",")
                    X22 = X2.split(",")
                    X33 = X3.split(",")
                    for k in range(len(X11)):
                            if Unique_Els[j] in X11[k] and len(Unique_Els[j]) <3:
                                if X22[k].isnumeric() :
                                    Output[i,j] = float(X22[k])/100
                                    Output[i,j+len(Unique_Els)] = float(X33[k])/100
                                # elif not X22[k] == X22[k]:
                                #     Output[i,j] = 100
                                #     Output[i,j+len(Unique_Els)] = 100
                elif "," not in X1 and len(Unique_Els[j]) <3 and Unique_Els[j] in X1 and X2.isnumeric():
                    
                    Output[i,j] = float(X2)/100
                    Output[i,j+len(Unique_Els)] = float(X3)/100
                # elif not X2.isnumeric():
                #     print(i)
                #     Output[i,0] = 100
                #     Output[i,0+len(Unique_Els)] = 100    
    for i in range(len(Array1)):
        if Output[i,:].sum() == 0:
            Output[i,0] = 1
            Output[i,0+len(Unique_Els)] = 1 
    return Output
        
        
def most_Occured(Unq_Title):
    Words = []
    for i in range(len(Unq_Title)):
        if Unq_Title[i] == Unq_Title[i]:
            wordList = re.sub("[^\w]", " ",  Unq_Title[i]).split()
            Words = Words + wordList
    
    Counter1 = Counter(Words)        
    most_occur = Counter1.most_common(700)
    most_occur_Titles = []
    for i in range(len(most_occur)):
        if (most_occur[i][1] > 30 and len(most_occur[i][0]) >= 3 and not most_occur[i][0] == "The"
            and not most_occur[i][0] == "the" and not most_occur[i][0] == "Web"
            and not most_occur[i][0] == "for" and not most_occur[i][0] == "and"
            and not most_occur[i][0] == "Your" and not most_occur[i][0] == "For"
            and not most_occur[i][0] == "www" and not most_occur[i][0] == "you"
            and not most_occur[i][0] == "Top" and not most_occur[i][0] == "all"
            and not most_occur[i][0] == "You" and not most_occur[i][0] == "net"
            and not most_occur[i][0] == "more" and not most_occur[i][0] == "San"
            and not most_occur[i][0] == "More" and not most_occur[i][0] == "Just"
            and not most_occur[i][0] == "One" and not most_occur[i][0] == "Get"
            and not most_occur[i][0] == "that" and not most_occur[i][0] == "How"
            and not most_occur[i][0] == "Real" and not most_occur[i][0] == "Gov"
            and not most_occur[i][0] == "our" and not most_occur[i][0] == "Home"
            and not most_occur[i][0] == "are" and not most_occur[i][0] == "since"
            and not most_occur[i][0] == "With" and not most_occur[i][0] == "What"
            and not most_occur[i][0] == "your" and not most_occur[i][0] == "new"
            and not most_occur[i][0] == "Our" and not most_occur[i][0] == "about"
            and not most_occur[i][0] == "All" and not most_occur[i][0] == "2021"
            and not most_occur[i][0] == "from" and not most_occur[i][0] == "And"
            and not most_occur[i][0] == "About" and not most_occur[i][0] == "High"
            and not most_occur[i][0] == "from" and not most_occur[i][0] == "And"):
            most_occur_Titles.append(most_occur[i][0])   
    return most_occur_Titles
            
import pandas as pd
import numpy as np
import re
import sys, os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

from SVM1 import MultiSVM
from daal.data_management import HomogenNumericTable

sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from customUtils import getArrayFromNT

data = pd.read_csv('Consumer_new.csv')
data["Category"]=data["Category"].str.strip()


corpus = []
tokens=[]
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['Description'][i])
    review = review.lower()
    review = word_tokenize(review)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    tokens=tokens+review
    review = ' '.join(review)
    corpus.append(review)
tokens=list(set(tokens))

di=[]

for i in tokens:
    temp=[]
    for j in range(len(corpus)):
        k=corpus[j].split()
        temp.append(k.count(i))
    di.append(temp)


ar=np.array(di)
ar=ar.T

print (len(corpus),len(tokens))
  
#desc=pd.DataFrame(columns=["Description"],data=ar)
#x=pd.concat(columns=[][desc,data["Category"]], axis=1)

y= data['Category'].factorize(0)

X_train, X_test, y_train, y_test = train_test_split(ar,y[0] , test_size = 0.2, random_state = 37)
print ("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))

trainData = HomogenNumericTable(X_train)
z = [[x] for x in y_train]
trainDependentVariables= HomogenNumericTable(z)
z = [[x] for x in y_test]
testData=HomogenNumericTable(X_test)
testGroundTruth = HomogenNumericTable(z)

daal_svm = MultiSVM(3,cacheSize=600000000)
#Train
trainingResult = daal_svm.training(trainData,trainDependentVariables)
#Predict
predictResults = daal_svm.predict(trainingResult,testData)

qualityMet = daal_svm.qualityMetrics(predictResults,testGroundTruth)
#print accuracy
print("Daal SVM Accuracy: "+ str(qualityMet.get('averageAccuracy')))

'''
----For testing----
do=[]
text="i want to know about my loan and home loan"
review1 = re.sub('[^a-zA-Z]', ' ', text)
review1 = review1.lower()
review1 = word_tokenize(review1)
reviews=""
ps = PorterStemmer()
for word in review1:
    if word in tokens:
        if word not in set(stopwords.words('english')):
            reviews = reviews+" "+ps.stem(word)
for i in tokens:
    temp=[]
    
    k=reviews.split()
    temp.append(k.count(i))
    do.append(temp)
            
iar=np.array(do)
iar=iar.T

#print (len(corpus),len(tokens))
reviewData = HomogenNumericTable(iar)
pred=daal_svm.predict(trainingResult,reviewData)

outcome={0:'Bank account or service',1:'Loan',2:'Credit card'}

l=getArrayFromNT(pred)

print(outcome[int(l[0][0])])

'''

import speech_recognition as sr 
import time

r = sr.Recognizer() 
mic_list = sr.Microphone.list_microphone_names() 
outcome={0:'Bank account or service',1:'Loan',2:'Credit card'}


pred=[]
def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        text = r.recognize_google(audio) 
        print( "you said: " + text )
        
        if "bye" in text:
            f=open('data.txt','a+')
            f.write("bye"+","+time.ctime()+","+"bye")
            sys.exit()
        review1 = re.sub('[^a-zA-Z]', ' ', text)
        review1 = review1.lower()
        review1 = word_tokenize(review1)
        reviews=""
        ps = PorterStemmer()
        for word in review1:
            if word in tokens:
                if word not in set(stopwords.words('english')):
                    reviews = reviews+" "+ps.stem(word)
        
        #review1 = ' '.join(review1)
        
        if reviews!="":
            do=[]
            for i in tokens:
                temp=[]
                
                k=reviews.split()
                temp.append(k.count(i))
                do.append(temp)
                        
            iar=np.array(do)
            iar=iar.T
            
            #print (len(corpus),len(tokens))
            reviewData = HomogenNumericTable(iar)
            pre=daal_svm.predict(trainingResult,reviewData)
            l=getArrayFromNT(pre)

            print(outcome[int(l[0][0])])
            pred.append([text,time.ctime(),outcome[int(l[0][0])]])
            
            f=open('data.txt','a+')
            f.write(text+","+time.ctime()+","+outcome[int(l[0][0])]+"\r\n")
            
            
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    

r = sr.Recognizer()
m = sr.Microphone(device_index=1)
with open('data.txt', 'w'): pass
stop_listening = r.listen_in_background(m, callback,phrase_time_limit=3)

# stop listening, wait for 5 seconds, then restart listening
stop_listening()
time.sleep(0.1)
stop_listening = r.listen_in_background(m, callback,phrase_time_limit=3)

print("start talking")
try:
    while True: time.sleep(0.1)

except KeyboardInterrupt:
        print("Exiting the call")

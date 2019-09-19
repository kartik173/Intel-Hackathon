import pandas as pd

import re
import sys
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize

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
tokens=set(tokens)    
desc=pd.DataFrame(columns=["Description"],data=corpus)
#x=pd.concat(columns=[][desc,data["Category"]], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(desc['Description'], data['Category'], test_size = 0.2, random_state = 37)
print ("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))


cv = CountVectorizer()
cv.fit(X_train)


X_train_cv = cv.transform(X_train)

X_test_cv = cv.transform(X_test)


mnb = MultinomialNB(alpha = 0.5)
mnb.fit(X_train_cv,y_train)

y_mnb = mnb.predict(X_test_cv)

#from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_cv,y_train)
y_mnb1 = classifier.predict(X_test_cv)


from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear', random_state = 0)
classifier1.fit(X_train_cv,y_train)
y_mnb2 = classifier1.predict(X_test_cv)

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier2.fit(X_train_cv,y_train)
y_mnb3 = classifier2.predict(X_test_cv)

print('LR: ', accuracy_score( y_mnb1 , y_test))
print('Naive Bayes Accuracy: ', accuracy_score( y_mnb , y_test))
print('SVM: ', accuracy_score( y_mnb2 , y_test))
print('RF: ', accuracy_score( y_mnb3 , y_test))


import speech_recognition as sr 
import time

r = sr.Recognizer() 
mic_list = sr.Microphone.list_microphone_names() 

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
            cv1 = CountVectorizer()
            rev=pd.Series(data=reviews)
            cv1.fit(rev)
        
            review1_cv = cv.transform(rev)
            pred.append([text,time.ctime(),classifier2.predict(review1_cv)[0]])
            
            f=open('data.txt','a+')
            f.write(text+","+time.ctime()+","+classifier2.predict(review1_cv)[0]+"\r\n")
            
            
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

# do other things on the main thread

print("start talking")
try:
    while True: time.sleep(0.1)

except KeyboardInterrupt:
        print("Exiting the call")
        








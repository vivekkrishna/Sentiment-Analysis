# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:12:36 2017

Sentiment analysis - NLP

@author: vc185059
"""
import os
import csv
import re

from sklearn.feature_extraction.text import CountVectorizer
corpus=[]
y=[]
with open('imdb_tr.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1]=='text':
            continue
        corpus.append(row[1])
        y+=row[2]
        
        
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)

#rows=len(vectorizer)
rows=len(corpus)
columns=len(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))

#finalX=X.toarray()

import scipy.sparse as sp
xArray = sp.lil_matrix((rows,columns),dtype=int)
xArray=X

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(xArray, y)

fo=open('unigram.output.txt','w')

testcorpus=[]
with open('imdb_te.csv', 'r',encoding = 'ISO-8859-1') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[1]=='text':
            continue
        testcorpus.append(row[1])

Xtopredict=vectorizer.transform(testcorpus)
xtopreArray = sp.lil_matrix((len(testcorpus),columns),dtype=int)
xtopreArray=Xtopredict
yout=clf.predict(xtopreArray)
for i in yout:
    fo.write(str(i))
    fo.write('\n')
fo.close()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(xArray)
xArray = sp.lil_matrix((rows,columns),dtype=int)
xArray=tfidf

clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(xArray, y)

fo=open('unigramtfidf.output.txt','w')

transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(Xtopredict)
xtopreArray = sp.lil_matrix((len(testcorpus),columns),dtype=int)
xtopreArray=tfidf

yout=clf.predict(xtopreArray)
for i in yout:
    fo.write(str(i))
    fo.write('\n')
fo.close()


#print(xArray[10,10])

bigram_vectorizer = CountVectorizer(ngram_range=(2,2),
                                    token_pattern=r'\b\w+\b', min_df=1)
#analyze = bigram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') == (
#    ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
X_2 = bigram_vectorizer.fit_transform(corpus)

columns=len(bigram_vectorizer.get_feature_names())

X_2Array= sp.lil_matrix((len(corpus),columns),dtype=int)

X_2Array=X_2

clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X_2Array, y)

fo=open('bigram.output.txt','w')

Xtopredict=bigram_vectorizer.transform(testcorpus)
xtopreArray = sp.lil_matrix((len(testcorpus),columns),dtype=int)
xtopreArray=Xtopredict
yout=clf.predict(xtopreArray)
for i in yout:
    fo.write(str(i))
    fo.write('\n')
fo.close()

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(X_2Array)
X_2Array = sp.lil_matrix((rows,columns),dtype=int)
X_2Array=tfidf

clf = SGDClassifier(loss="hinge", penalty="l1")
clf.fit(X_2Array, y)

fo=open('bigramtfidf.output.txt','w')

transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(xtopreArray)
xtopreArray = sp.lil_matrix((len(testcorpus),columns),dtype=int)
xtopreArray=tfidf

yout=clf.predict(xtopreArray)
for i in yout:
    fo.write(str(i))
    fo.write('\n')
fo.close()


"""
writing to csv

"""

fstopwords=open('stopwords.en.txt','r')

stopwords=set()

for word in fstopwords:
    fword=word.strip('\n')
    stopwords.add(fword)

stopwords.add('br')
query = 'What is hello'

fw=open('imdb_tr.csv','w', newline='')
al = csv.writer(fw, delimiter=',')
data = [['','text','polarity']]
al.writerows(data)
index=0
directory = os.path.normpath("C:\\Users\\vc185059\\Desktop\\Lstudy\\AI\\NLP\\SENTIMENT ANALYSIS\\aclImdb\\train\\pos")
for subdir, dirs, files in os.walk(directory):
#    for diri in dirs:
#        print(diri)
#        if diri=='pos' or diri=='neg':
    for file in files:
        #print(file)
        if file.endswith(".txt"):
            l=file.split('.')
            li=l[0].split('_')
            if int(li[1])>5:
                polarity=1
            else:
                polarity=0
            f=open(os.path.join(subdir, file),'r')
            try:
                a = f.read()
            except UnicodeDecodeError:
                continue
            
            #a=str(a)
            a.strip('<br /><br />')
            
            querywords=re.split('\W+', a)
            #print(querywords)
            resultwords  = [word for word in querywords if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            #print(result)
            result.strip('br')
            al = csv.writer(fw, delimiter=',')
            data = [[str(index),result,str(polarity)]]
            al.writerows(data)
            index+=1
            f.close()
            
directory = os.path.normpath("C:\\Users\\vc185059\\Desktop\\Lstudy\\AI\\NLP\\SENTIMENT ANALYSIS\\aclImdb\\train\\neg")
for subdir, dirs, files in os.walk(directory):
#    for diri in dirs:
#        print(diri)
#        if diri=='pos' or diri=='neg':
    for file in files:
        #print(file)
        if file.endswith(".txt"):
            l=file.split('.')
            li=l[0].split('_')
            if int(li[1])>5:
                polarity=1
            else:
                polarity=0
            f=open(os.path.join(subdir, file),'r')
            try:
                a = f.read()
            except UnicodeDecodeError:
                continue
            
            #a=str(a)
            a.strip('<br /><br />')
            a.replace('<br /><br />','')
            #querywords = a.split()
            querywords=re.split('\W+', a)
            #print(querywords)
            resultwords  = [word for word in querywords if word.lower() not in stopwords]
            result = ' '.join(resultwords)
            #print(result)
            result.strip('br')
            al = csv.writer(fw, delimiter=',')
            data = [[str(index),result,str(polarity)]]
            al.writerows(data)
            index+=1
            f.close()
                      
fw.close()


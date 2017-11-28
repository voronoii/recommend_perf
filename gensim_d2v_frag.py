from __future__ import print_function

import gensim
import os
import collections
import smart_open
import random
from gensim import corpora, models, similarities
from collections import defaultdict
import csv
from gensim.matutils import jaccard_distance as jd
from gensim.models.keyedvectors import KeyedVectors
import codecs
global dictt
import time
import signal
import sys
from gensim.models.doc2vec import LabeledSentence
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import mpld3
import multiprocessing
from nltk.tokenize import RegexpTokenizer
from nltk.metrics.distance import jaccard_distance
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


dictt = {}

#csv 파일을 라인별(문서별)로 읽어서 id를 매김
#리뷰를 담고 있는 각 문서가 좀더 맥주의 특색을 담기 위해 해당 스타일을 문서에 추가함.
def read_corpus(filename):
    f = open(filename, 'r')
    rdr = csv.reader(f)

    # word_list = []
    for idx, line in enumerate(rdr):
        name = line[0].strip()
        words = line[1].split()
        dictt[name] = words

        #문서마다 태그 부여
        yield gensim.models.doc2vec.TaggedDocument(words, tags=[name])
        words = []
        # count = 0




def makemodel():

    train_corpus = list(read_corpus('./cleaned_words_edited.csv'))
    cores = multiprocessing.cpu_count()
    vec_size = 300
    model = gensim.models.doc2vec.Doc2Vec(dm=1, dm_concat=0, iter = 20, size=vec_size, window=5, negative=1, hs=0, min_count=1, workers=cores)
    model.build_vocab(train_corpus)    
    model.intersect_word2vec_format('D:/GoogleNews-vectors-negative300.bin', lockf=1.0, binary=True)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter, start_alpha=model.alpha , end_alpha=model.alpha*0.8179)
    model.save('./doc2vec(using cleaned csv, lock1.0, dm)2.d2v')

    train_corpus = list(read_corpus('./cleaned_words_edited.csv'))
    cores = multiprocessing.cpu_count()
    vec_size = 300
    model = gensim.models.doc2vec.Doc2Vec(dm=0, dm_concat=0, iter = 20, size=vec_size, window=5, negative=1, hs=0, min_count=1, workers=cores)
    model.build_vocab(train_corpus)    
    model.intersect_word2vec_format('D:/GoogleNews-vectors-negative300.bin', lockf=1.0, binary=True)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter, start_alpha=model.alpha , end_alpha=model.alpha*0.8179)
    model.save('./doc2vec(using cleaned csv, lock1.0, dbow)2.d2v')

    
    
def recommend_items(infer_vector):
    ##infer vector 단어가 많을수록 퍼센티지가 올라가는 경향이 있음.

    # f = open('./recomment_result.csv','w', newline='')
    # wr  = csv.writer(f)
    result = []
    try:             
        
        train_corpus = list(read_corpus('./cleaned_words.csv'))
        # model = gensim.models.doc2vec.Doc2Vec.load('./doc2vec(using cleaned csv, lock1.0, dbow).d2v')
        model = gensim.models.doc2vec.Doc2Vec.load('/scentpincette/doc2vec(using cleaned csv, lock1.0, dm).d2v')

        
        ## method 1
        inferred_vector = model.infer_vector(infer_vector)
        sims = model.docvecs.most_similar([inferred_vector], topn=1)
        for sim in sims:
            result.append(sim[0])
            
        
        ## method 2
        infer_vector_ = infer_vector + infer_vector + infer_vector    
        inferred_vector = model.infer_vector(infer_vector_)
        sims2 = model.docvecs.most_similar([inferred_vector], topn=2)
        
        for sim in sims2:
            result.append(sim[0])

        ## method 3        
        additional_words = model.wv.most_similar(infer_vector, topn=4)  
        infer_vector_ = infer_vector  
        for ele in additional_words:
            infer_vector_.append(ele[0])
     
        inferred_vector = model.infer_vector(infer_vector_)
        sims3 = model.docvecs.most_similar([inferred_vector], topn=2)
        

        for sim in sims3:
            result.append(sim[0])

        

    except KeyError:
        pass

    return result

    # f.close()



def test(exam_list):
    print("testing....")

    f = open('testresult.csv','w',newline='')
    wr = csv.writer(f)

    result = [] 
    for ele in exam_list:

        infer_vector = ele.split()

        try:             
            
            train_corpus = list(read_corpus('./cleaned_words.csv'))
            # model = gensim.models.doc2vec.Doc2Vec.load('./doc2vec(using cleaned csv, lock1.0, dbow).d2v')
            model = gensim.models.doc2vec.Doc2Vec.load('D:/opinionmining/doc2vec(using cleaned csv, lock1.0, dm).d2v')

            
            ## method 1
            inferred_vector = model.infer_vector(infer_vector)
            sims = model.docvecs.most_similar([inferred_vector], topn=1)
            for sim in sims:
                wr.writerow([sim[0],sim[1]])
                print(sim[0], sim[1])
                
            
            ## method 2
            infer_vector_ = infer_vector + infer_vector + infer_vector    
            inferred_vector = model.infer_vector(infer_vector_)
            sims2 = model.docvecs.most_similar([inferred_vector], topn=2)
            print("--------------------------------------")
            for sim in sims2:
                wr.writerow([sim[0],sim[1]])
                print(sim[0], sim[1])

            ## method 3        
            additional_words = model.wv.most_similar(infer_vector, topn=4)  
            infer_vector_ = infer_vector  
            for ele in additional_words:
                infer_vector_.append(ele[0])
         
            inferred_vector = model.infer_vector(infer_vector_)
            sims3 = model.docvecs.most_similar([inferred_vector], topn=2)
            print("----------------------------------------")

            for sim in sims3:
                wr.writerow([sim[0],sim[1]])
                print(sim[0], sim[1])

            
              
        except KeyError:
            print("except")


        f.close() 
        return result


from gensim.models.keyedvectors import KeyedVectors

def test_corpus():  
    
    model = gensim.models.doc2vec.Doc2Vec.load('D:/opinionmining/doc2vec(using cleaned csv, lock1.0, dm).d2v')
    result = model.wv.most_similar(positive=['toffee'])
    print(result)

import math
def Cosine(vec1, vec2) :
    result = InnerProduct(vec1,vec2) / (VectorSize(vec1) * VectorSize(vec2))
    return result

def VectorSize(vec) :
    return math.sqrt(sum(math.pow(v,2) for v in vec))

def InnerProduct(vec1, vec2) :
    return sum(v1*v2 for v1,v2 in zip(vec1,vec2))

def Euclidean(vec1, vec2) :
    return math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in zip(vec1, vec2)))

def Theta(vec1, vec2) :
    return math.acos(Cosine(vec1,vec2)) + 10

def Triangle(vec1, vec2) :
    theta = math.radians(Theta(vec1,vec2))
    return (VectorSize(vec1) * VectorSize(vec2) * math.sin(theta)) / 2

def Magnitude_Difference(vec1, vec2) :
    return abs(VectorSize(vec1) - VectorSize(vec2))

def Sector(vec1, vec2) :
    ED = Euclidean(vec1, vec2)
    MD = Magnitude_Difference(vec1, vec2)
    theta = Theta(vec1, vec2)
    return math.pi * math.pow((ED+MD),2) * theta/360    

def TS_SS(vec1, vec2):
    return Triangle(vec1, vec2) * Sector(vec1, vec2)

import operator
def tsssTest():
    infer_vector = ['orange','citrus','woody']
    model = gensim.models.doc2vec.Doc2Vec.load('D:/opinionmining/doc2vec(using cleaned csv, lock1.0, dm).d2v')
    item_label_dict = {}

    docvec_list = list(model.docvecs)
    index = 0
    result_dict = {}

    for name in model.docvecs.doctags.keys():
        item_label_dict[name] = docvec_list[index]
        index+=1

    inferred_vector = model.infer_vector(infer_vector)
    for key, value in item_label_dict.items():
        
        result = TS_SS(value, inferred_vector)
        result_dict[key] = result

    sorted_dict = sorted(result_dict.items(), key=operator.itemgetter(1))

    index = 0

    print(sorted_dict[len(sorted_dict)-4:len(sorted_dict)])

def main():
    tsssTest()


if __name__ == '__main__':
    main()  
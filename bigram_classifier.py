import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()

from nltk.tokenize import RegexpTokenizer
tknzr = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stoplist = set(stopwords.words('english'))
stoplist.update(['\n', ' ', "'", ])
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import os
import gensim
from collections import namedtuple
import numpy as np
from bs4 import BeautifulSoup
from nltk.classify import NaiveBayesClassifier
from unidecode import unidecode

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from random import shuffle
import nltk.classify.util, nltk.metrics

from gensim.models import Doc2Vec
import multiprocessing
from collections import Counter
import operator
import csv
import re

global moviedict, num_topics, tot_titles, bestwords
moviedict = {}
testdict = {}
num_topics = 0
tot_titles = ""
bestwords = ""


class GenSimCorpus(object):
       def __init__(self, texts, stoplist=[],bestwords=[],stem=False):
           self.texts = texts
           self.stoplist = stoplist
           self.stem = stem
           self.bestwords = bestwords
           self.dictionary = gensim.corpora.Dictionary(self.iter_docs(texts, stoplist))
        
       def __len__(self):
           return len(self.texts)
       def __iter__(self):
           for tokens in self.iter_docs(self.texts, self.stoplist):
               yield self.dictionary.doc2bow(tokens)
       def iter_docs(self,texts, stoplist):
           for text in texts:
               if self.stem:
                  yield (stemmer.stem(w) for w in [x for x in tknzr.tokenize(text) if x not in stoplist])
               else:
                  if len(self.bestwords)>0:
                     yield (x for x in tknzr.tokenize(text) if x in self.bestwords)
                  else:
                     yield (x for x in tknzr.tokenize(text) if x not in stoplist)  


def ListDocs(dirname):
        docs = []
        titles = []
        for filename in [f for f in os.listdir(dirname) if str(f)[0]!='.']:
            f = open(dirname+'/'+filename,'r')
            id = filename.split('.')[0].split('_')[1]
            titles.append(moviedict[id])
            docs.append(f.read())
        return docs,titles

def GenerateDistrArrays(corpus, num_topics):
         for i,dist in enumerate(corpus[:10]):
             dist_array = np.zeros(num_topics)
             for d in dist:
                 dist_array[d[0]] =d[1]
             # if dist_array.argmax() == 6 :
             #    print (tot_titles[i])


######필터링##########
def PreprocessReviews(text,stop=[],stem=False):
    
        words = tknzr.tokenize(text)
        if stem:
           words_clean = [stemmer.stem(w) for w in [i.lower() for i in words if i not in stop]]
        else:
           words_clean = [i.lower() for i in words if i not in stop]

       
        return words_clean

def word_features(words):
    return dict([(word, True) for word in words])

def bigrams_words_features(words, nbigrams,measure=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(measure, nbigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def bigrams_words_features2(words, nbigrams):
    bigram_finder = BigramCollocationFinder.from_words(words)

    ##모두 더함 
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, nbigrams)
    bigrams += bigram_finder.nbest(BigramAssocMeasures.pmi, nbigrams)
    bigrams += bigram_finder.nbest(BigramAssocMeasures.mi_like, nbigrams)
    bigrams += bigram_finder.nbest(BigramAssocMeasures.jaccard, nbigrams)

    
    # open('bigrams4.txt', 'w').write('\n'.join('%s %s' % x for x in bigrams)) ## bigrams=리스트 안에 바이그램 튜플들 
    
    ###네가지 방법 중 3개이상 방법에서 연음이라고 한 것만을 연음으로 인정한다.
    collocations = [ tup for tup, f in Counter(bigrams).most_common() if f >= 3]
    
    return dict([(ngram, True) for ngram in itertools.chain(words, collocations)])
      


def best_words_features(words):
    return dict([(word, True) for word in words if word in bestwords])


def main():

	print("processing...........")

    ######### moviedict 만들기 #################
	moviehtmldir = 'D:\opinionmining\polarity_html\movie'
    

	for filename in [f for f in os.listdir(moviehtmldir) if f[0]!='.']:

		id = filename.split('.')[0]
        
		f = open(moviehtmldir+'/'+filename, 'r', encoding='utf8', errors='replace')  
        
		parsed_html = BeautifulSoup(f.read(), 'lxml')        
		try:
		    title = unidecode(parsed_html.body.h1.text)

		except:
		    title = 'none'
		moviedict[id] = title
	print("training dictionary has been made.........")
	print()





###########데이터 사전 처리##########################
	Review = namedtuple('Review','words title tags')

	dir = './review_polarity/txt_sentoken/'
	doc2vecstem = True
	reviews_pos = []
	cnt = 0
	for filename in [f for f in os.listdir(dir+'pos/') if str(f)[0]!='.']:
	    f = open(dir+'pos/'+filename,'r')
	    id = filename.split('.')[0].split('_')[1]
	    reviews_pos.append(Review(PreprocessReviews(f.read(),stoplist,doc2vecstem),moviedict[id],['pos_'+str(cnt)]))
	    cnt+=1
        
    

	reviews_neg = []
	cnt= 0
	for filename in [f for f in os.listdir(dir+'neg/') if str(f)[0]!='.']:
	    f = open(dir+'neg/'+filename,'r')
	    id = filename.split('.')[0].split('_')[1]
	    reviews_neg.append(Review(PreprocessReviews(f.read(),stoplist,doc2vecstem),moviedict[id],['neg_'+str(cnt)]))
	    cnt+=1



	tot_reviews = reviews_pos + reviews_neg

	############# twitter 데이터 처리 ###################
	
	# f= open('./twitter/twitter_neg.csv', 'r')
	# rdr = csv.reader(f)

	# negfeatures_twitter = []
	# for idx, comment in enumerate(rdr):
	# 	com = comment[1]
	# 	negfeatures_twitter.append((bigrams_words_features2(com,50), 'neg'))


	# ff= open('./twitter/twitter_pos.csv', 'r')
	# rdrr = csv.reader(ff)

	# posfeatures_twitter = []
	# for idx, comment in enumerate(rdrr):
	# 	com = comment[1]
	# 	posfeatures_twitter.append((bigrams_words_features2(com,50), 'pos'))



	# #############bigram 훈련#########################
	negfeatures = [(bigrams_words_features(r.words,500), 'neg') for r in reviews_neg]
	posfeatures = [(bigrams_words_features(r.words,500), 'pos') for r in reviews_pos]
	portionpos = int(len(posfeatures)*0.8)
	portionneg = int(len(negfeatures)*0.8)


	trainfeatures = negfeatures[:portionpos] + posfeatures[:portionneg] 											
	testfeatures = negfeatures[portionpos:] + posfeatures[portionneg:]  
	classifier = NaiveBayesClassifier.train(trainfeatures)




	# ######### testdata를 위한 dict만들기 ############
	# testdir = 'D:/opinionmining/testfor_separate/'

	# for fname in os.listdir(testdir):
	#     path = os.path.join(testdir, fname)
	#     if os.path.isdir(path):
	#         continue
	#     else:
	#         id = fname.split('-')[0]    #id랑 이름이랑 같게될듯 
	#         f = open(testdir+'/'+fname, 'r', encoding='utf8', errors='replace')
	#         try:
	#             title = unidecode(fname)
	       
	#         except:
	#             title = 'none'
	#         testdict[id] = title
	# print("testing dictionary has been made.........")
	# print() 

	# tests = []
	# cnt= 0
	# for fname in os.listdir(testdir):
	#     path = os.path.join(testdir, fname)
	#     if os.path.isdir(path):
	#         continue
	#     else:
	#         f = open(testdir+fname,'r')
	#         id = fname.split('-')[0]
	#         tests.append(Review(PreprocessReviews(f.read(),stoplist,doc2vecstem),testdict[id],['']))
	#         cnt+=1

	# testfeatures_ = [(bigrams_words_features2(r.words,500), r.title) for r in tests]      
	# print("testfeatures has been made.........")


	############ classifier test ###############
	err = 0
	print( 'test on: ',len(testfeatures))
	for r in testfeatures:
		print(r[0])
		sent = classifier.classify(r[0])
	    
	    #print r[1],'-pred: ',sent
		if sent != r[1]:
	   		err +=1.
	print("classifier test ----------------")
	print ('error rate: ',err/float(len(testfeatures)))  


	############ classification ###############
	# fff = open('result2.csv', 'w', newline='')
	# rdrr = csv.writer(fff)
	# for r in testfeatures_:
	#     sent = classifier.classify(r[0])
	#     rdrr.writerow([r[1], sent])

	# fff.close()
	# print("classification has done.")

	import _pickle as cPickle
	filename = 'finalized_model.sav'
	cPickle.dump(classifier, open(filename, 'wb'))

	


if __name__ == '__main__':
	main()
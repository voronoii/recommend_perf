import _pickle as cPickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()

from nltk.tokenize import RegexpTokenizer
tknzr = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stoplist = set(stopwords.words('english'))
stoplist.update(['\n', ' ', "'",'days','found','find','fragrance.','first','little','reviews','review','notes.','slowly','could','years','edt','edp','completely','original', 'skin','felt','feels','feel','wear','much',"i've",'get','lasts','last','would','bottle','note','notes','like','this','one','attract','attraction','the','scent','ever','smell','scent','fragrance','fragrances',"don't",'eau','and','nice','oud','code',"it's","i'm",'perfume','perfumes','edition','love','also','new','even','quite','way','good','well','really','sure','definitely','slightly','almost','scent.','buy','bought','take','took','taken','got','cologne','growing','make','makes','made','something','anything','nothing','smells','think','thinks','thought',"can't",
	'need','needs','updated','opinion','shop','taken','literally','never','review','read','life','ever','gives','deal','smelling','smelled','perfumes','smells','amazing','price','store','early','features','notes',
'note','fragrance','out','rating','votes','cologne',
'thank','thanks','well','favorite','pours','since','nice','body','perfume', 'small','with','brand',
'would','great','scent','ml','bottle','like','much','less','l', 'head', 'commercial', 'description','ingredients','kind','give','away','expert','probably','explore',
'scent','skin','body','hour','day','sometimes','good','nothing','self','best','find','apply','wonderfully','feel','miss','add','blend','taken',
'holds','hold','might','even','wrong','attempt','often', 'really','launch', 'launched', 'behind',
'gorgeous', 'love', 'face', 'boyfriend', 'another', 'still', 'thing', 'alot', 'sample', 'throw', 'collection','person','election','went',
'online', 'order','ordered','immediately','loved', 'surprisingly','think','later','hour','hours','incredible','would','something','smell','come','year','make','buy','bought','though',
'spray','dozen','quite','little','base','note','take','took','taken','gave','definitely','clothes','better','reminds',
'strange','agree','color','childhood','meant','mean','meaning','saying','sale','produce','tell', 'feeling','dont'])
stoplist.update(open('./company.txt').read().split())
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import itertools
import csv
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from random import shuffle
import nltk.classify.util, nltk.metrics
from collections import Counter
from nltk.probability import FreqDist
import re
doc2vecstem = True




def classifier():
	reviewdir = 'D:/opinionmining/frag_reviews/'
	classifier = cPickle.load(open('./finalized_model.sav', 'rb'))
	print("processing....................")

	for fname in os.listdir(reviewdir):
	    path = os.path.join(reviewdir, fname)
	    f = open(path, 'r')
	    review_list = list(f.read().split('\n\n'))      
	    Review = []
	    review_for_test = []
	    filtered = []

### review_for_test = [ [Review], [Review], [Review].....]
	    for para in review_list:    
	    	Review.append(para)
	    	Review.append(PreprocessReviews(para,stoplist,doc2vecstem))
	    	review_for_test.append(Review)
	    	Review = []

	    try:
	    	testfeatures = [(bigrams_words_features2(review[1],500), review[0]) for review in review_for_test]
	    except:
	    	pass

	    for x in testfeatures:
	    	sent = classifier.classify(x[0])
	    	print(sent)
	    	if sent != 'neg' :
	    		filtered.append(x[1])

	    ff = open('./frag_reviews_classified/'+fname+'.txt', 'w')
	    ff.write(' '.join(filtered))
	    ff.close()

	    filtered = []
	    testfeatures = []
	    review_for_test = []






def PreprocessReviews(text,stop=[],stem=False):
    
        words = tknzr.tokenize(text)
        if stem:
           words_clean = [stemmer.stem(w) for w in [i.lower() for i in words if i not in stop]]
        else:
           words_clean = [i.lower() for i in words if i not in stop]

       
        return words_clean

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

def bigrams_words_features(words, nbigrams,measure=BigramAssocMeasures.chi_sq):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(measure, nbigrams)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def most_common():

	classified_dir = 'D:/opinionmining/frag_reviews_classified/'
	ff = open('./filtered_words.csv','w', newline='')
	wr = csv.writer(ff)
	for fname in os.listdir(classified_dir):
	    path = os.path.join(classified_dir, fname)

	    f = open(path, 'r')
	    text = re.sub('[^A-Za-z0-9 ]+', '', f.read())
	    temp = list(text.lower().split())

	    stoplist.update(fname.split())
	    filtered_words = [w for w in temp if not w in stoplist and len(w)>2]

	    fdist = FreqDist(filtered_words)
	    words = fdist.most_common(15)
	    common_words = ""
	    for idx, word in enumerate(words):
	        common_words += (words[idx][0]+" ")
	    wr.writerow([fname, common_words])
	ff.close()

	    
def word_clean():

	f = open('./filtered_words.csv','r')
	rdr = csv.reader(f)

	ff = open('./cleaned_words.csv','w',newline='')
	wr = csv.writer(ff)
	
	for idx, line in enumerate(rdr):
		word_list = [word for word in line[1].split() if len(word) > 2]
		string = ""
		for word in word_list:
			string += (word+' ')
		print(word_list)
		print()
		wr.writerow([line[0], string])

	ff.close()
	f.close()

def file_word_clean():
	reviewdir = 'D:/opinionmining/frag_reviews_classified/'
	for fname in os.listdir(reviewdir):
	    path = os.path.join(reviewdir, fname)
	    f = open(path, 'r')
	    words = tknzr.tokenize(f.read().lower())
	    
	    filtered = [word for word in words if word not in stoplist and len(word) > 2]
	    string = ""
	    for token in filtered : 
	    	string += token+" "

	    f = open('D:/opinionmining/frag_reviews_classified2/'+fname, 'w')
	    f.write(string)
	    f.close()

def main():
	file_word_clean()



if __name__ == '__main__':
	main()
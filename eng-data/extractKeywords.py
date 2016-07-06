from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import re
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
stop = stopwords.words('english')


class LemmaTokenizer(object):
	def __init__(self):
		#self.wnl = WordNetLemmatizer()
		self.stemmer = LancasterStemmer()
	def __call__(self, doc):
		#return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if re.match(r'[a-z]+', t, re.M|re.I)]
		return [self.stemmer.stem(t) for t in word_tokenize(doc) if re.match(r'[a-z]+', t, re.M|re.I)]


vectorizer = TfidfVectorizer(min_df=50, tokenizer=LemmaTokenizer(),stop_words=stop)

corpus = []

with open("./test.csv") as csv_test, open("./train.csv") as csv_train:
	reader_test = csv.reader(csv_test, delimiter=',', quotechar='"')
	reader_train = csv.reader(csv_train, delimiter=',', quotechar='"')
	
	for row in reader_test:
		corpus.append(row[2])
	for row in reader_train:
		corpus.append(row[2])


X = vectorizer.fit_transform(corpus)

with open("./words.txt", 'wb') as f:
	print len(vectorizer.get_feature_names())
	for word in vectorizer.get_feature_names():
		f.write(word+"\n")


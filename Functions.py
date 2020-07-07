# -*- coding: utf-8 -*-
from os import listdir
import os
import sys
import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import operator
import string
import codecs
from nltk.tag import pos_tag_sents
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from nltk import ngrams
from nltk.corpus import wordnet
# this example uses TopicRank
import matplotlib.pyplot as plt
import numpy as np
import argparse
from gensim.models import KeyedVectors
import xml.etree.ElementTree as ET
from nltk.collocations import *
import sys
from sklearn.manifold import TSNE
import os.path
from subprocess import call
from itertools import chain
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from collections import Counter
# from sklearn.ensemble import IsolationForest
from nltk.collocations import *
from sklearn import manifold
import matplotlib.font_manager
from sklearn.decomposition import IncrementalPCA
import json

ps = PorterStemmer()

class Functions(object):
 
    def __init__(self, file_path, file_name):
        self.file_name = file_name
        self.file_path = file_path
        self.keyphrases = []

    def document_gold_keyphrases(self, keyphrases_path): 
          # readers_and_authors=True in case we search for the intersection of the authors'
          # and readers' keyphrases
          with open(keyphrases_path) as json_file:
            data = json.load(json_file)
          list_of_lists_keyphrases = data[self.file_name.replace('.xml', '').replace('.txt', '')]
          for list_keyphrase in list_of_lists_keyphrases:
            self.keyphrases.append(list_keyphrase[0].lower().translate(str.maketrans('', '', string.punctuation)).split())  
          return self.keyphrases

    # def document_gold_keyphrases_lemmas(self, keyphrases_path): 
    #       # readers_and_authors=True in case we search for the intersection of the authors'
    #       # and readers' keyphrases
    #       lemmatizer = WordNetLemmatizer()
    #       with open(keyphrases_path) as json_file:
    #         data = json.load(json_file)
    #       list_of_lists_keyphrases = data[self.file_name.replace('.xml', '')]
    #       for list_keyphrase in list_of_lists_keyphrases:
    #         words = list_keyphrase[0].split()
    #         words_lemmas = []
    #         for word in words:
    #           words_lemmas.append(lemmatizer.lemmatize(word))
    #         self.keyphrases.append(words_lemmas)  
    #       return self.keyphrases

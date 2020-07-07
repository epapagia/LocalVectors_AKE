# -*- coding: utf-8 -*-

"""Base classes for the pke module."""

from collections import defaultdict

from pke.data_structures import Candidate, Document
from pke.readers import MinimalCoreNLPReader, RawTextReader

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import RegexpParser
from nltk.corpus import stopwords
from nltk.tag.mapping import map_tag
from nltk.util import ngrams
from nltk.collocations import *

from string import punctuation
import os
import logging
import codecs
import math
import string

import spacy

import re

from six import string_types

from builtins import str

import xml.etree.ElementTree as ET

import numpy as np
from numpy import linalg

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics.pairwise import cosine_similarity

import operator

import networkx as nx

from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
import scipy

import sys

ISO_to_language = {'en': 'english', 'pt': 'portuguese', 'fr': 'french',
                   'es': 'spanish', 'it': 'italian', 'nl': 'dutch',
                   'de': 'german'}

escaped_punctuation = {'-lrb-': '(', '-rrb-': ')', '-lsb-': '[', '-rsb-': ']',
                       '-lcb-': '{', '-rcb-': '}'}


class LoadFile(object):
    """The LoadFile class that provides base functions."""

    def __init__(self, input_file):
        """Initializer for LoadFile class."""

        self.input_file = input_file
        """Path to the input file."""

        self.language = None
        """Language of the input file."""

        self.normalization = None
        """Word normalization method."""

        self.sentences = []
        """Sentence container (list of Sentence objects)."""

        self.candidates = defaultdict(Candidate)
        """Keyphrase candidates container (dict of Candidate objects)."""

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self._models = os.path.join(os.path.dirname(__file__), 'models')
        """Root path of the models."""

        self._df_counts = os.path.join(self._models, "df-semeval2010.tsv.gz")
        """Path to the document frequency counts provided in pke."""

        self.stoplist = None
        """List of stopwords."""

        self.clean_sentences = {}
        
        self.dict_sentid_bigrams = {}

        self.dict_sentid_trigrams = {}
        
        self.dict_realSentId_matrixSentId = {}

        self.clean_sentences_unstemmed = {}

        self.document_tokens = []

        self.vocab = {}

        self.sorted_outliers = []
#---------------------------------------------------------
        self.sorted_outlier_sentences = []

        self.candidates_outliers = {}

        self.document_local_vector = np.zeros(shape=(1, 1))

        self.document_words = []

        self.candidates_local_vectors = {}

        self.document_vector = np.zeros(shape=(1, 1))

        self.candidates_vectors = {}

        self.dict_bigramsText_pmi = {}

        self.dict_trigramsText_pmi = {}

        self.graph = nx.Graph()

        self.weights = {}
        """Weight container (can be either word or candidate weights)."""

        self.weights_tfidf = {}

        self.weights_mah_rank = {}

        self.sorted_candidate_keywords_mah_rank = []

        self.vocab_phrases = {}

        self.sorted_inliers = []

        self.graph_local_vecs = nx.Graph()

        self.weights_tfidf = {}

        self.dict_cooccurrences = {}

        self.cooccurrences_matrix = np.zeros(shape=(1, 1))

        self.sorted_outliers_glove = []

        self.sorted_candidate_keywords_mah_rank_glove = []

        self.sorted_outliers_cooccurrences = []

        self.sorted_candidate_keywords_mah_rank_cooccurrences = []

#----------------------------------functions for MCD tuning--------------------------------------------
    def tune_RM(self, Wpos, W, ivocab, score_type = 'custom', support_fraction = 'default', measure = 'mahalanobis'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        if support_fraction == 'default':
          support_fraction = ((W.shape[0]+W.shape[1])/2.0)/W.shape[0]
        else:
          support_fraction = support_fraction

        clf = MinCovDet(assume_centered=False, support_fraction=support_fraction)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_
        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        
        self.sorted_outliers = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

        return self.sorted_outliers

    
    def tune_RM_cooccurrences(self, Wpos, W, ivocab, score_type = 'custom', support_fraction = 'default', measure = 'mahalanobis'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """
        
        if support_fraction == 'default':
          support_fraction = ((W.shape[0]+W.shape[1])/2.0)/W.shape[0]
        else:
          support_fraction = support_fraction

        clf = MinCovDet(assume_centered=False, support_fraction=support_fraction)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        
        self.sorted_outliers_cooccurrences = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

        
        return self.sorted_outliers_cooccurrences



    def tune_RM_glove(self, Wpos, vocab_pos, W, ivocab, score_type = 'custom', support_fraction = 'default', measure = 'mahalanobis'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        if support_fraction == 'default':
          support_fraction = ((W.shape[0]+W.shape[1])/2.0)/W.shape[0]
        else:
          support_fraction = support_fraction

        clf = MinCovDet(assume_centered=False, support_fraction=support_fraction)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)  

        
        self.sorted_outliers_glove = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)


        return self.sorted_outliers_glove

#----------------------------------functions for RM tuning--------------------------------------------

#----------------------------------functions for SC tuning--------------------------------------------
    def tune_SC(self, Wpos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        mean_vector = np.mean(W, axis=0)
        W_cov = np.cov(W.T)

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        
        self.sorted_outliers = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

        return self.sorted_outliers


    def tune_SC_cooccurrences(self, Wpos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """
        mean_vector = np.mean(W, axis=0)
        W_cov = np.cov(W.T)

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        
        self.sorted_outliers_cooccurrences = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

        
        return self.sorted_outliers_cooccurrences



    def tune_SC_glove(self, Wpos, vocab_pos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """
        mean_vector = np.mean(W, axis=0)
        W_cov = np.cov(W.T)

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)  

        
        self.sorted_outliers_glove = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)


        return self.sorted_outliers_glove

#----------------------------------functions for SC tuning--------------------------------------------

#----------------------------------functions for EC tuning--------------------------------------------
    def tune_EC(self, Wpos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        clf = EmpiricalCovariance(assume_centered=False)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_
        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        
        self.sorted_outliers = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)

        return self.sorted_outliers
    
    def tune_EC_cooccurrences(self, Wpos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        clf = EmpiricalCovariance(assume_centered=False)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)  
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[i, :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)

        self.sorted_outliers_cooccurrences = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)        
        return self.sorted_outliers_cooccurrences

    def tune_EC_glove(self, Wpos, vocab_pos, W, ivocab, score_type = 'custom', measure = 'euclidean'):
        """Produce local word embedding (GloVe).
        Args:
        Wpos (numpy 2d array): the initial word-sentence position_matrix before PCA (required to count word occurrences in document)
        W (numpy 2d array): the vectors (rows: the different vectors, cols: the vector dimensions).
        """

        clf = EmpiricalCovariance(assume_centered=False)
        clf.fit(W)
        mean_vector = clf.location_
        # W_cov = np.cov(W.T)
        W_cov = clf.covariance_

        W_cov_inv = np.zeros((1,1))
        if linalg.cond(W_cov) < 1/sys.float_info.epsilon:
          W_cov_inv = linalg.inv(W_cov)

        dist = {}
        if measure == 'mahalanobis':
          if W_cov_inv.shape[1] > 1:
            if score_type == 'custom':
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
            else:
              for i in range(0, W.shape[0]):
                dist[ivocab[i]] = mahalanobis(W[i,:], mean_vector, W_cov_inv)   
          else:
            print('W is invertible!')
            return self.sorted_outliers
        elif measure == 'euclidean':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = euclidean(W[i,:], mean_vector)
        elif measure == 'cosine':
          if score_type == 'custom':
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)*(1.0/(np.nonzero(Wpos[vocab_pos[ivocab[i]], :])[0][0] + 1.0))    
          else:
            for i in range(0, W.shape[0]):
              dist[ivocab[i]] = cosine(W[i,:], mean_vector)  

        self.sorted_outliers_glove = sorted(dist.items(), key=operator.itemgetter(1), reverse=True)
        return self.sorted_outliers_glove

#----------------------------------functions for EC tuning--------------------------------------------

    def build_cooccurrences(self, window=10):
        """Build a co-occurrrence matrix among the document's words.
            Sentence boundaries **are not** taken into account in the
            window.
            Args:
                window (int): the window for counting the co-occurrence between two words,
                    defaults to 10.
        """
        text = []
        for sentence in self.sentences:
            for i, word in enumerate(sentence.stems):
                text.append(word)

        for i, word1 in enumerate(text[:int(len(text))]):
            if word1 in self.vocab.keys():
                if word1 not in self.dict_cooccurrences.keys():
                    self.dict_cooccurrences[word1] = {}
                for j in range(i + 1, min(i + window, len(text[:int(len(text))]))):
                    word2 = text[j]
                    if word1 != word2 and word2 in self.vocab.keys():
                        if word2 not in self.dict_cooccurrences[word1].keys():
                            self.dict_cooccurrences[word1][word2] = 1.0
                        else:
                            self.dict_cooccurrences[word1][word2] += 1.0
        return self.dict_cooccurrences

    def get_cooccurrences_matrix(self):
        """Convert co-occurrences dictionary to a co-occurrrence numpy matrix among the document's words.
    """
        self.cooccurrences_matrix = np.zeros(shape=(len(self.vocab.keys()), len(self.vocab.keys())))  # len(self.vocab.keys())))
        for w1 in self.dict_cooccurrences.keys():
            for w2 in self.dict_cooccurrences[w1].keys():
                self.cooccurrences_matrix[self.vocab[w1], self.vocab[w2]] = self.dict_cooccurrences[w1][w2]
        return self.cooccurrences_matrix

    def build_graph(self, window=10, pos=None):
        """Build a graph representation of the document in which nodes/vertices
        are words and edges represent co-occurrence relation. Syntactic filters
        can be applied to select only words of certain Part-of-Speech.
        Co-occurrence relations can be controlled using the distance (window)
        between word occurrences in the document.

        The number of times two words co-occur in a window is encoded as *edge
        weights*. Sentence boundaries **are not** taken into account in the
        window.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 10.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
        """
        text = []
        if pos is None:
          # flatten document as a sequence of (word, pass_syntactic_filter) tuples
          text = [(word, word in self.vocab.keys()) for sentence in self.sentences
                  for i, word in enumerate(sentence.stems)]
        else:
          # flatten document as a sequence of (word, pass_syntactic_filter) tuples
          text = [(word, word in self.vocab.keys() and sentence.pos[i] in pos) for sentence in self.sentences
                  for i, word in enumerate(sentence.stems)] #sentence.pos[i] in pos and word not in stopwords and len(word)>1
                                                            # 
        # add nodes to the graph
        self.graph.add_nodes_from([word for word, valid in text if valid])

        # add edges to the graph
        for i, (node1, is_in_graph1) in enumerate(text):

            # speed up things
            if not is_in_graph1:
                continue

            for j in range(i + 1, min(i + window, len(text))):
                node2, is_in_graph2 = text[j]
                if is_in_graph2 and node1 != node2:
                    if not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(node1, node2, weight=0.0)
                    self.graph[node1][node2]['weight'] += 1.0
        
        return len(self.graph.nodes)

    def get_candidates(self):
        return self.candidates

    def pca_projection(self, cooccurrences_matrix, num_components):
        pca = PCA(n_components=num_components)
        pca_cooccurrences_matrix = pca.fit_transform(cooccurrences_matrix)
        return pca_cooccurrences_matrix

    def ngram_position_matrix(self):
        
        ind = 0
        # print(self.clean_sentences)
        for key_id, value_text in enumerate(self.clean_sentences.items()):
            for token in self.clean_sentences[key_id][0]:
                if token not in self.vocab.keys():
                    self.vocab[token] = ind
                    ind = ind + 1
        
        ivocab = {v: k for k, v in self.vocab.items()}
        position_matrix = np.zeros(shape=(len(self.vocab.keys()), len(self.clean_sentences.keys())))
        
        for sent_id, sent in self.clean_sentences.items():
            for token_id, token in enumerate(sent[0]):
                position_matrix[self.vocab[token], sent_id] = token_id + 1
          
        return position_matrix, self.vocab, ivocab

    def get_clean_sentence(self, stopwords_list, common_adjectives, reporting_verbs,
                           determiners, functional_words):
        lemmatizer = WordNetLemmatizer()
        sent_id_updated = 0
        for sent_id, sentence in enumerate(self.sentences):
            # self.sentences[sent_id].stems = sentence.words
            sent_tokens_clean_stemmed_lower = []
            sent_tokens_clean_unstemmed_lower = []
            dict_token_PosTag = {}
            for word_id, word in enumerate(self.sentences[sent_id].words):
                word_text = re.sub('[^A-Za-z0-9-]+', '', str(word)).lower()
                if str(word_text) not in stopwords_list and lemmatizer.lemmatize(
                        str(word_text)) not in common_adjectives and \
                        lemmatizer.lemmatize(str(word_text)) not in reporting_verbs and lemmatizer.lemmatize(
                    str(word_text)) not in determiners and \
                        lemmatizer.lemmatize(str(word_text)) not in functional_words and not str(word_text).replace(
                    ".", "").replace(",", "").isdigit() and len(lemmatizer.lemmatize(str(word_text).replace(".", "").replace(",", "").replace("-", ""))) > 1:
                    sent_tokens_clean_stemmed_lower.append(re.sub('[^A-Za-z0-9-]+', '', str(self.sentences[sent_id].stems[word_id])))#lemmatizer.lemmatize(str(word_text)))) #str(self.sentences[sent_id].stems[word_id]))
                    sent_tokens_clean_unstemmed_lower.append(word.lower())
                    dict_token_PosTag[len(sent_tokens_clean_stemmed_lower) - 1] = (
                        self.sentences[sent_id].pos[word_id], self.sentences[sent_id].tags[word_id])
            if len(sent_tokens_clean_stemmed_lower) > 0:
                self.dict_realSentId_matrixSentId[sent_id] = sent_id_updated
                bigrams = ngrams(sent_tokens_clean_stemmed_lower, 2)
                dict_bi_PosTag = {}
                for i in range(len(sent_tokens_clean_stemmed_lower) - 1):
                    dict_bi_PosTag[i] = (dict_token_PosTag[i], dict_token_PosTag[i + 1])
                trigrams = ngrams(sent_tokens_clean_stemmed_lower, 3)
                dict_tri_PosTag = {}
                for i in range(len(sent_tokens_clean_stemmed_lower) - 2):
                    dict_tri_PosTag[i] = (dict_token_PosTag[i], dict_token_PosTag[i + 1], dict_token_PosTag[i + 2])
                self.clean_sentences[self.dict_realSentId_matrixSentId[sent_id]] = (
                    sent_tokens_clean_stemmed_lower, dict_token_PosTag)
                self.clean_sentences_unstemmed[self.dict_realSentId_matrixSentId[sent_id]] = (
                    sent_tokens_clean_unstemmed_lower, dict_token_PosTag)
                self.dict_sentid_bigrams[self.dict_realSentId_matrixSentId[sent_id]] = (list(bigrams), dict_bi_PosTag)
                self.dict_sentid_trigrams[self.dict_realSentId_matrixSentId[sent_id]] = (
                    list(trigrams), dict_tri_PosTag)
                self.document_tokens.extend(sent_tokens_clean_stemmed_lower)
                sent_id_updated += 1
        
        return self.clean_sentences, self.dict_sentid_bigrams, self.dict_sentid_trigrams, self.dict_realSentId_matrixSentId
    
    def read_document_from_xml_file(self):
          title_clean_doc_text = ''
          # parse an xml file by name
          tree = ET.parse(self.input_file)
          root = tree.getroot()
          for document in root.findall('document'):
              for sentences in document.findall('sentences'):
                  for sentence in sentences.findall('sentence'):
                      for tokens in sentence.findall('tokens'):
                          for token in tokens.findall('token'):
                              for word in token.findall('word'):
                                  title_clean_doc_text += word.text + ' '
                      title_clean_doc_text += '\n'
          title_clean_doc_text = title_clean_doc_text.strip()
          return title_clean_doc_text

    def load_document(self, input, **kwargs):
        """Loads the content of a document/string/stream in a given language.

        Args:
            input (str): input.
            language (str): language of the input, defaults to 'en'.
            encoding (str): encoding of the raw file.
            normalization (str): word normalization method, defaults to
                'stemming'. Other possible values are 'lemmatization' or 'None'
                for using word surface forms instead of stems/lemmas.
        """

        # get the language parameter
        language = kwargs.get('language', 'en')

        # test whether the language is known, otherwise fall back to english
        if language not in ISO_to_language:
            logging.warning(
                "ISO 639 code {} is not supported, switching to 'en'.".format(
                    language))
            language = 'en'

        # initialize document
        doc = Document()

        if isinstance(input, string_types):

            # if input is an input file
            if os.path.isfile(input):

                # an xml file is considered as a CoreNLP document
                if input.endswith('xml'):
                    parser = MinimalCoreNLPReader()
                    doc = parser.read(path=input, **kwargs)
                    doc.is_corenlp_file = True

                # other extensions are considered as raw text
                else:
                    parser = RawTextReader(language=language)
                    encoding = kwargs.get('encoding', 'utf-8')
                    with codecs.open(input, 'r', encoding=encoding) as file:
                        text = file.read()
                    doc = parser.read(text=text, path=input, **kwargs)

            # if input is a string
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input, **kwargs)

        elif getattr(input, 'read', None):
            # check whether it is a compressed CoreNLP document
            name = getattr(input, 'name', None)
            if name and name.endswith('xml'):
                parser = MinimalCoreNLPReader()
                doc = parser.read(path=input, **kwargs)
                doc.is_corenlp_file = True
            else:
                parser = RawTextReader(language=language)
                doc = parser.read(text=input.read(), **kwargs)

        else:
            logging.error('Cannot process {}'.format(type(input)))

        # set the input file
        self.input_file = doc.input_file

        # set the language of the document
        self.language = language

        # set the sentences
        self.sentences = doc.sentences

        # initialize the stoplist
        self.stoplist = stopwords.words(ISO_to_language[self.language])

        # word normalization
        self.normalization = kwargs.get('normalization', 'stemming')#'lemmatization') #stemming
        if self.normalization == 'stemming':
            self.apply_stemming()
        elif self.normalization is None:
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].stems = sentence.words

        # lowercase the normalized words
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [w.lower() for w in sentence.stems]

        # POS normalization
        if getattr(doc, 'is_corenlp_file', False):
            self.normalize_pos_tags()
            self.unescape_punctuation_marks()

    def apply_stemming(self):
        """Populates the stem containers of sentences."""

        if self.language == 'en':
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer("porter") #WordNetLemmatizer()
        else:
            # create a new instance of a porter stemmer
            stemmer = SnowballStemmer(ISO_to_language[self.language],
                                      ignore_stopwords=True)

        # iterate throughout the sentences
        for i, sentence in enumerate(self.sentences):
            self.sentences[i].stems = [stemmer.stem(w) for w in sentence.words] #lemmatize

    def normalize_pos_tags(self):
        """Normalizes the PoS tags from udp-penn to UD."""

        if self.language == 'en':
            # iterate throughout the sentences
            for i, sentence in enumerate(self.sentences):
                self.sentences[i].pos = [map_tag('en-ptb', 'universal', tag)
                                         for tag in sentence.pos]

    def unescape_punctuation_marks(self):
        """Replaces the special punctuation marks produced by CoreNLP."""

        for i, sentence in enumerate(self.sentences):
            for j, word in enumerate(sentence.words):
                l_word = word.lower()
                self.sentences[i].words[j] = escaped_punctuation.get(l_word,
                                                                     word)

    def is_redundant(self, candidate, prev, minimum_length=1):
        """Test if one candidate is redundant with respect to a list of already
        selected candidates. A candidate is considered redundant if it is
        included in another candidate that is ranked higher in the list.

        Args:
            candidate (str): the lexical form of the candidate.
            prev (list): the list of already selected candidates (lexical
                forms).
            minimum_length (int): minimum length (in words) of the candidate
                to be considered, defaults to 1.
        """

        # get the tokenized lexical form from the candidate
        candidate = self.candidates[candidate].lexical_form

        # only consider candidate greater than one word
        if len(candidate) < minimum_length:
            return False

        # get the tokenized lexical forms from the selected candidates
        prev = [self.candidates[u].lexical_form for u in prev]

        # loop through the already selected candidates
        for prev_candidate in prev:
            for i in range(len(prev_candidate) - len(candidate) + 1):
                if candidate == prev_candidate[i:i + len(candidate)]:
                    return True
        return False

    def get_n_best(self, n=10, redundancy_removal=False, stemming=False):
        """Returns the n-best candidates given the weights.

        Args:
            n (int): the number of candidates, defaults to 10.
            redundancy_removal (bool): whether redundant keyphrases are
                filtered out from the n-best list, defaults to False.
            stemming (bool): whether to extract stems or surface forms
                (lowercased, first occurring form of candidate), default to
                False.
        """

        # sort candidates by descending weight
        best = sorted(self.weights, key=self.weights.get, reverse=True)

        # remove redundant candidates
        if redundancy_removal:

            # initialize a new container for non redundant candidates
            non_redundant_best = []

            # loop through the best candidates
            for candidate in best:

                # test wether candidate is redundant
                if self.is_redundant(candidate, non_redundant_best):
                    continue

                # add the candidate otherwise
                non_redundant_best.append(candidate)

                # break computation if the n-best are found
                if len(non_redundant_best) >= n:
                    break

            # copy non redundant candidates in best container
            best = non_redundant_best

        # get the list of best candidates as (lexical form, weight) tuples
        n_best = [(u, self.weights[u]) for u in best[:min(n, len(best))]]

        # replace with surface forms if no stemming
        if not stemming:
            n_best = [(' '.join(self.candidates[u].surface_forms[0]).lower(),
                       self.weights[u]) for u in best[:min(n, len(best))]]

        if len(n_best) < n:
            logging.warning(
                'Not enough candidates to choose from '
                '({} requested, {} given)'.format(n, len(n_best)))

        # return the list of best candidates
        return n_best

    def add_candidate(self, words, stems, pos, offset, sentence_id):
        """Add a keyphrase candidate to the candidates container.

        Args:
            words (list): the words (surface form) of the candidate.
            stems (list): the stemmed words of the candidate.
            pos (list): the Part-Of-Speeches of the words in the candidate.
            offset (int): the offset of the first word of the candidate.
            sentence_id (int): the sentence id of the candidate.
        """

        # build the lexical (canonical) form of the candidate using stems
        lexical_form = ' '.join(stems)

        # add/update the surface forms
        self.candidates[lexical_form].surface_forms.append(words)

        # add/update the lexical_form
        self.candidates[lexical_form].lexical_form = stems

        # add/update the POS patterns
        self.candidates[lexical_form].pos_patterns.append(pos)

        # add/update the offsets
        self.candidates[lexical_form].offsets.append(offset)

        # add/update the sentence ids
        self.candidates[lexical_form].sentence_ids.append(sentence_id)

    def ngram_selection(self, n=3):
        """Select all the n-grams and populate the candidate container.

        Args:
            n (int): the n-gram length, defaults to 3.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # limit the maximum n for short sentence
            skip = min(n, sentence.length)

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # generate the ngrams
            for j in range(sentence.length):
                for k in range(j + 1, min(j + 1 + skip, sentence.length + 1)):
                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[j:k],
                                       stems=sentence.stems[j:k],
                                       pos=sentence.pos[j:k],
                                       offset=shift + j,
                                       sentence_id=i)

    def longest_pos_sequence_selection(self, valid_pos=None):
        self.longest_sequence_selection(
            key=lambda s: s.pos, valid_values=valid_pos)

    def longest_keyword_sequence_selection(self, keywords):
        self.longest_sequence_selection(
            key=lambda s: s.stems, valid_values=keywords)

    def longest_sequence_selection(self, key, valid_values):
        """Select the longest sequences of given POS tags as candidates.

        Args:
            key (func) : function that given a sentence return an iterable
            valid_values (set): the set of valid values, defaults to None.
        """

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # container for the sequence (defined as list of offsets)
            seq = []

            # loop through the tokens
            for j, value in enumerate(key(self.sentences[i])):

                # add candidate offset in sequence and continue if not last word
                if value in valid_values:
                    seq.append(j)
                    if j < (sentence.length - 1):
                        continue

                # add sequence as candidate if non empty
                if seq:

                    # add the ngram to the candidate container
                    self.add_candidate(words=sentence.words[seq[0]:seq[-1] + 1],
                                       stems=sentence.stems[seq[0]:seq[-1] + 1],
                                       pos=sentence.pos[seq[0]:seq[-1] + 1],
                                       offset=shift + seq[0],
                                       sentence_id=i)

                # flush sequence container
                seq = []

    def grammar_selection(self, grammar=None):
        """Select candidates using nltk RegexpParser with a grammar defining
        noun phrases (NP).

        Args:
            grammar (str): grammar defining POS patterns of NPs.
        """

        # initialize default grammar if none provided
        if grammar is None:
            grammar = r"""
                NBAR:
                    {<NOUN|PROPN|ADJ>*<NOUN|PROPN>} 
                    
                NP:
                    {<NBAR>}
                    {<NBAR><ADP><NBAR>}
            """

        # initialize chunker
        chunker = RegexpParser(grammar)

        # loop through the sentences
        for i, sentence in enumerate(self.sentences):

            # compute the offset shift for the sentence
            shift = sum([s.length for s in self.sentences[0:i]])

            # convert sentence as list of (offset, pos) tuples
            tuples = [(str(j), sentence.pos[j]) for j in range(sentence.length)]

            # parse sentence
            tree = chunker.parse(tuples)

            # find candidates
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    leaves = subtree.leaves()

                    # get the first and last offset of the current candidate
                    first = int(leaves[0][0])
                    last = int(leaves[-1][0])

                    # add the NP to the candidate container
                    self.add_candidate(words=sentence.words[first:last + 1],
                                       stems=sentence.stems[first:last + 1],
                                       pos=sentence.pos[first:last + 1],
                                       offset=shift + first,
                                       sentence_id=i)

    @staticmethod
    def _is_alphanum(word, valid_punctuation_marks='-'):
        """Check if a word is valid, i.e. it contains only alpha-numeric
        characters and valid punctuation marks.

        Args:
            word (string): a word.
            valid_punctuation_marks (str): punctuation marks that are valid
                    for a candidate, defaults to '-'.
        """
        for punct in valid_punctuation_marks.split():
            word = word.replace(punct, '')
        return word.isalnum()

    def candidate_filtering(self,
                            stoplist=None,
                            minimum_length=3,
                            minimum_word_size=2,
                            valid_punctuation_marks='-',
                            maximum_word_number=5,
                            only_alphanum=True,
                            pos_blacklist=None):
        """Filter the candidates containing strings from the stoplist. Only
        keep the candidates containing alpha-numeric characters (if the
        non_latin_filter is set to True) and those length exceeds a given
        number of characters.
            
        Args:
            stoplist (list): list of strings, defaults to None.
            minimum_length (int): minimum number of characters for a
                candidate, defaults to 3.
            minimum_word_size (int): minimum number of characters for a
                token to be considered as a valid word, defaults to 2.
            valid_punctuation_marks (str): punctuation marks that are valid
                for a candidate, defaults to '-'.
            maximum_word_number (int): maximum length in words of the
                candidate, defaults to 5.
            only_alphanum (bool): filter candidates containing non (latin)
                alpha-numeric characters, defaults to True.
            pos_blacklist (list): list of unwanted Part-Of-Speeches in
                candidates, defaults to [].
        """

        if stoplist is None:
            stoplist = []

        if pos_blacklist is None:
            pos_blacklist = []

        # loop through the candidates
        for k in list(self.candidates):

            # get the candidate
            v = self.candidates[k]

            # get the words from the first occurring surface form
            words = [u.lower() for u in v.surface_forms[0]]

            # discard if words are in the stoplist
            if set(words).intersection(stoplist):
                del self.candidates[k]

            # discard if tags are in the pos_blacklist
            elif set(v.pos_patterns[0]).intersection(pos_blacklist):
                del self.candidates[k]

            # discard if containing tokens composed of only punctuation
            elif any([set(u).issubset(set(punctuation)) for u in words]):
                del self.candidates[k]

            # discard candidates composed of 1-2 characters
            elif len(''.join(words)) < minimum_length:
                del self.candidates[k]

            # discard candidates containing small words (1-character)
            elif min([len(u) for u in words]) < minimum_word_size:
                del self.candidates[k]

            # discard candidates composed of more than 5 words
            elif len(v.lexical_form) > maximum_word_number:
                del self.candidates[k]

            # discard if not containing only alpha-numeric characters
            if only_alphanum and k in self.candidates:
                if not all([self._is_alphanum(w, valid_punctuation_marks)
                            for w in words]):
                    del self.candidates[k]


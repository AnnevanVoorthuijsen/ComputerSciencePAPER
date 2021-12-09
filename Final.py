# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 08:56:42 2021

@author: Frederique
"""

from scipy.cluster.hierarchy import dendrogram, linkage
import json
import numpy as np
import random
import time
import pandas as pd
import math as math
import itertools as itertools
import scipy as scipy
import sklearn as sklearn
from sklearn.cluster import AgglomerativeClustering

with open(r'TVs-all-merged.json') as Jsonfile:
        data = json.load(Jsonfile)
with open(r'TVs-all-merged1.json') as Jsonfile:
        data1 = json.load(Jsonfile)

temp_list=[]
for key in data:
    temp_list.append(key)

titles_1 = []
titles = ''
keys=[]
#create list of all titles
for j in range(len(temp_list)):
        for item in data1['TVs']:
            for name in item[temp_list[j]]:         #hier zit je in elke j
                title = name['title']
                key= name['modelID']
                titles_1.append(title)
                titles = (titles + ' ' + title)           #dit moet mis nu anders
                keys.append(key)

X = titles.split()
MW = list(dict.fromkeys(X))     #create VECTOR OF MODEL WORDS


def preprocessing(titles_1, keys):
    def bootstrap(titles_1, keys):
        length=len(titles_1)
        total = [*range(0, length, 1)]
   
        training = random.choices(total, weights=None, cum_weights=None, k=length)
        training = list( dict.fromkeys(training) )           #remove all duplicates
        training = list( dict.fromkeys(training) )
   
        test = list(set(total) - set(training))              #list with all numbers not in training

        #CREATE JSON FOR BOTH DATASETS

        testset1=[]
        testset2=[]
        trainingset1=[]
        trainingset2=[]
        for i in range(0, length):
            if i in test:
                testset1.append(keys[i])
                testset2.append(titles_1[i])
            else:
                trainingset1.append(keys[i])
                trainingset2.append(titles_1[i])
    
        return ([testset1, testset2], [trainingset1, trainingset2])
    
    (testset, trainingset)=bootstrap(titles_1, keys)
    
    #Maak titles en MW
    def binaryvectors(keys, titles_1): 
        titles=''
        for i in range(0, len(titles_1)):
            titles = (titles + ' ' + titles_1[i])
            
        X = titles.split()
        MW = list(dict.fromkeys(X))  

        #create binary columns
        k = 0
        b = np.zeros((len(MW),len(titles_1)))
        for i in titles_1:
            Y = i.split()
            for j in range(len(MW)):
                for x in Y:
                    if MW[j] == x:
                        b[j, k] = 1
            k = k + 1
        return b, MW


    test_b=binaryvectors(testset[0], testset[1])
    training_b=binaryvectors(trainingset[0], trainingset[1])

    def createhash(b, MW, titles_1):
        #MINHASING
        #create permutations
        random.seed(49)
        p1 = random.sample(range(len(MW)), len(MW))
        p2 = random.sample(range(len(MW)), len(MW))
        p3 = random.sample(range(len(MW)), len(MW))
        p4 = random.sample(range(len(MW)), len(MW))
        p5 = random.sample(range(len(MW)), len(MW))
        p6 = random.sample(range(len(MW)), len(MW))
        p7 = random.sample(range(len(MW)), len(MW))
        p8 = random.sample(range(len(MW)), len(MW))
        p = [p1,p2, p3, p4, p5, p6, p7, p8]  

        def create_hash(vector: list):      #hash per product and create list
            signature = []                  #create empty list
            for j in range(0,8):            # for elke permutation
                for i in range(1, len(MW)): #voor elk woord in de model words vector
                    idx = p[j].index(i)         #inx is de index van de permutation
                    signature_val = vector[idx]     #sign value is die index in de input (is b)
                    if signature_val == 1:      # als die value 1 is, dan is dat de value voor de signature matrix.
                        signature.append(i)             #toevoegen aan signature vector. misschien i+1, want hij print nu index en wil je niet liever gewoon de echte plaats?
                        break
            return signature

        length=len(titles_1)                             #make of all the vectors one matrix
        signature2 = np.zeros((8,len(titles_1)))             #empty signature matrix
        for x in range(0, len(titles_1)):
            signature2[:,x] = create_hash(b[:,x])     #run de fuctie create_hash voor elke column van b en maak er 1 matrix van.
        #signature_T = np.transpose(signature2)
        #jaccard similarity
        #def jaccard(a: set, b: set):
         #   return len(a.intersection(b)) / len(a.union(b))
        return(temp_list, keys, titles_1, signature2)

    trainingsignature=createhash(training_b[0], training_b[1], trainingset[1])
    testsignature=createhash(test_b[0], test_b[1], testset[1])

    return (trainingsignature, testsignature)

trainingsignature1 = preprocessing(titles_1, keys)[0]
testsignature2 = preprocessing(titles_1, keys)[1]

def LSH_complete(signature2):
    def splitsignature(signature, r):  # de arrays hier hebben opeens floats.
        b = int(len(signature) / r)  # hier splitsen we de signature in meerdere bands
        subvector = []
        for i in range(0, len(signature), r):
            subvector.append(signature[i: i + r])
        return subvector

    def LSH(signature, r):
        buckets = {}
        length=len(signature[1])
        for j in range(0, length, 1):  # voor alle signatures (producten)
            bandj = splitsignature(signature[:, j], r)  # maak de band
            for i in range(0, len(bandj)):  # loop over de hoeveelheid bands (hier 2)
                band = bandj[i]
                bandedj = ""
                for t in range(0, len(band)):
                    X = band[t]
                    bandedj = bandedj + str(X)[:-2]
                if bandedj in buckets.keys():
                    buckets[bandedj].append(j)
                else:
                    buckets.setdefault(bandedj, [])
                    integer_to_append = j
                    buckets[bandedj].append(integer_to_append)
        key_list_buckets = []
        
        for key in buckets:
            key_list_buckets.append(key)
                   
        z=0           
        candidate_pairs=[]
        for i in range(0, len(key_list_buckets)):
            length=len(buckets[key_list_buckets[i]])
            if length<=1:
                next
            else:
                Matches= list(itertools.combinations(buckets[key_list_buckets[i]], 2))
                for j in range(0, len(Matches)):
                    Match=str(Matches[j]) 
                    candidate_pairs.append(Match)
                    z+=1
        
        candidate_pairs = list(dict.fromkeys(candidate_pairs))
        return candidate_pairs, key_list_buckets

    buckets = LSH(signature2, 4)
    return buckets

LSH_train = LSH_complete(trainingsignature1[3])
LSH_test = LSH_complete(testsignature2[3])



potential_duplicates_train = LSH_train[0]
potential_duplicates_test = LSH_test[0]

key_list_buckets_train = LSH_train[1]
key_list_buckets_test = LSH_test[1]

#START MSM
theta1 = 0.2
theta2 = 0.3
mu = 0.5
def MSM(theta1, theta2, mu, candidate_pairs, key_list_buckets, data1, temp_list, signature2):
    titles_1 = []
    titles = ''
    keys = []
    # create list of all titles and keys
    for j in range(len(temp_list)):
        for item in data1['TVs']:
            for key in item:
                keys.append(key)
            for name in item[temp_list[j]]:
                title = name['title']
                titles_1.append(title)
                titles = (titles + title)

    def duplicates_in_list(x, y):
        count = 0
        for num in y:
            if num in x:
                count += 1
        return count

    def qgrams(a, b):  # calculates how similar two strings are
        elements_a = []  # create lists for the 3 characteristics
        elements_b = []
        for i in range(0, len(a) - 2):
            A = a.replace(' ', '')  # delete spaces
            B = b.replace(' ', '')
            elements_a.append(A[i:i + 3])  # q-grams neem elke keer 3 letters en stop die in een list
            elements_b.append(B[i:i + 3])
        dup = duplicates_in_list(elements_a, elements_b)  # calculate number of duplicates between the two lists
        dice_coef = 2 * dup / (len(elements_a) + len(elements_b))  # calculate dice coeff
        return dice_coef

    def dissimilarity_matrix_func(theta1, theta2, mu, candidate_pairs, key_list_buckets, data1, temp_list, signature2):
        dissimilarity_matrix = np.ones((len(signature2[1]), len(signature2[1]))) * 100000
        for t in range(0, len(candidate_pairs)):  # loop over elk pair
            candidatepair = candidate_pairs[t]
            candidatepair = candidatepair.split(", ")
            hsim = 0
            candidatepair_1 = candidatepair[0].split("(")
            candidatepair_2 = candidatepair[1].split(")")
            candidatepair1 = int(candidatepair_1[1])
            candidatepair2 = int(candidatepair_2[0])
            SIM = qgrams(key_list_buckets[candidatepair1], key_list_buckets[candidatepair2])  # caculate how similar two keys of the main dictorionary are.
            if SIM > 0.1:  # alleen meenemen als groter dan bepaalde waarde (0.1 is random)
                hsim = theta1 * SIM
            for item in data1['TVs']:  # calculate how many similar model words in title.
                for name in item[temp_list[candidatepair1]]:
                    X = name['title']
                    X.split()  # get individual words of the titles
                for name in item[temp_list[candidatepair2]]:
                    Y = name['title']
                    Y.split()
                num_dup = duplicates_in_list(X, Y)
                perc_matching_mw = num_dup / (len(X) + len(Y))  # eigenlijk is dit hetzelfde als q gram
                hsim = hsim + theta2 * perc_matching_mw  # add to the hsim with weight theta2
            for item in data1['TVs']:  # doe q gram op de title zonder spaties.
                for name in item[keys[candidatepair1]]:
                    X = name['title']
                    X.replace(' ', '')
                for name in item[keys[candidatepair2]]:
                    Y = name['title']
                    Y.replace(' ', '')
                SIM2 = qgrams(X, Y)
                hsim = hsim + mu * SIM2
            idx = candidatepair1  # get indices of the products (hier twijfel ik beetje over)
            idy = candidatepair2
            dissimilarity_matrix[idx, idy] = 1 - hsim  # voeg toe aan de matrix
        return dissimilarity_matrix

    dissimilarity_matrix = dissimilarity_matrix_func(theta1, theta2, mu, candidate_pairs, key_list_buckets, data1, temp_list, signature2)

    epsilon = 0.52
    clusters = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single', distance_threshold=epsilon)
    clustered = clusters.fit(dissimilarity_matrix)
    cluster = clusters.fit_predict(dissimilarity_matrix)

    dictionary = {}
    for i in range(len(cluster)):
        if cluster[i] in dictionary:
            dictionary[cluster[i]].append(i)
        else:
            dictionary.setdefault(cluster[i], [])
            dictionary[cluster[i]].append(i)

    finalpairs = {}
    z = 0
    test_keys = []
    for i in dictionary:
        length = len(dictionary[i])
        if length <= 1:
            next
        else:
            Matches = list(itertools.combinations(dictionary[i], 2))
            test_keys.append(key)
            for j in range(0, len(Matches)):
                finalpairs[z] = Matches[j]
                z += 1

    #for i in dictionary:
     #   if len(dictionary[i]) > 1:
      #      print(dictionary[i]);

    return dictionary, finalpairs, test_keys

MSM_train = MSM(theta1, theta2, mu, potential_duplicates_train, key_list_buckets_train, data1, temp_list, trainingsignature1)
MSM_test = MSM(theta1, theta2, mu, potential_duplicates_test, key_list_buckets_test, data1, temp_list, testsignature2)
finalpairs_train = MSM_train[1]


def output(test_keys, found_pairs):
    def findtrueduplicates(keys):       #gaat hier niet helemaal lekker
        print('x')
        trueMatches = {}
        finalduplicates = {}
        length = len(keys)
        for i in range(0, length):
            if keys[i] in trueMatches.keys():
                trueMatches[keys[i]].append(i)
            else:
                trueMatches.setdefault(keys[i], [])
                trueMatches[keys[i]].append(i)
        z = 0
        print(trueMatches)
        for i in trueMatches:
            length = len(trueMatches[i])
            print(trueMatches[i])
            print(length)
            if length <= 1:
                print('x')
                next
            else:
                print('y')
                possiblematches = int((length * (length - 1)) / 2)
                Matches = list(itertools.combinations(trueMatches[i], 2))
                for j in range(0, possiblematches):
                    finalduplicates[z] = Matches[j]
                    print(z)
                    z += 1
        return finalduplicates

    def F1_score(found_pairs, true_duplicates):
        truepositive = 0
        falsepositive = 0
        for i in range(0, len(found_pairs)):
            for j in range(0, len(true_duplicates)):
                duplicated = found_pairs[i].sort()
                trueduplicate = true_duplicates(j).sort()
                if duplicated == trueduplicate:
                    truepositive += 1
                else:
                    falsepositive += 1
        falsenegative = len(true_duplicates) - len(found_pairs)

        F1_score = truepositive / (truepositive + 0.5 * (falsepositive + falsenegative))
        print(F1_score)
        return (F1_score, truepositive)

    true_duplicates = findtrueduplicates(test_keys)
    F_1 = F1_score(found_pairs, true_duplicates)

    return F_1[0], F_1[1], true_duplicates

output_model = output(temp_list, finalpairs_train)

r = 4
repeats = 5
F1_scoreavg = []
lentrue = []
amountfound_total = []
amountcompared = []

def MSM_optimization(trainingdata, train_keys):
    parameters = []

    grid_parameters = {
        "theta_1": range(0, 1, 0.1),  # epsilon nog toevoegen
        "theta_2": range(0, 1, 0.1),
        "mu": range(0, 1, 0.1)
    }

    # minimize F_1 over theta1, theta2, mu
    model = sklearn.model_selection.RandomizedSearchCV(MSM, grid_parameters,
                                                       scoring=output(train_keys, found_pairs)[0]).fit(trainingdata)
    # ik twijfel heeel erg of dit werkt, maar op hoop van zegen maar runnen.

    parameters = model.best_params_  # vm wordt dit een dictionary, heeft invloed op de loop hieronder
    return parameters


for i in range(0, repeats):
    (testsignature, trainsignature, test_keys, train_keys) = preprocessing(titles_1, keys)
    potentialduplicates_test = LSH_complete(testsignature, r)
    potentialduplicates_train = LSH_complete(trainsignature, r)

    train_parameters = MSM_optimization(trainingdata, train_keys)
    test_duplicatesfound = MSM(test, train_parameters[0], train_parameters[1], train_parameters[2])

    (F_1, amountfound, amounttrue) = output(test_keys, test_duplicatesfound)

    F1_scoreavg.append(F_1)
    lentrue.append(amounttrue)
    amountfound_total.append(amountfound)
    amountcompared.append(len(potentialduplicates_test))

F1 = np.mean(F1_scoreavg)

print(F1);


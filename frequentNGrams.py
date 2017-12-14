import csv
import os
import sys
import re

from nltk import FreqDist

from collections import Counter

BASE_DIR = ''
GENRE_DIR = os.path.join(BASE_DIR, "singlegenre")

def ngrams(n, words):
    gram = dict()
    for i in range(len(words)-(n-1)):
        if n == 1:
            key = (words[i])
        elif n == 2:
            key = (words[i], words[i+1])
        elif n == 3:
            key = (words[i], words[i+1], words[i+2])
        else:
            print("Sorry, can't support more than 3-gram")
        
        if key in gram:
            gram[key] += 1
        else:
            gram[key] = 1

    # Turn into a list of (word, count) sorted by count from most to least
    gram = sorted(gram.items(), key=lambda words: words[1], reverse = True)
    return gram
    

for name in sorted(os.listdir(GENRE_DIR)):
    path = os.path.join(GENRE_DIR, name)
    lyricsList = []

    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                t = re.sub('[^a-z\ \']+', " ", t)
                lyricsList.extend(t.split())
                f.close()

    print(name)
    unigrams = ngrams(1, lyricsList)
    bigrams = ngrams(2, lyricsList)
    trigrams = ngrams(3, lyricsList)
    # for i in range (0,10):
    #     print(str(unigrams[i][0]) + ": " + str(unigrams[i][1]))
    # for i in range(0,10):
    #     print(str(bigrams[i][0][0]) + " " + str(bigrams[i][0][1]) + ": " + str(bigrams[i][1]))
    for i in range(0,10):
        print(str(trigrams[i][0][0]) + " " + str(trigrams[i][0][1]) + " " + str(trigrams[i][0][2]) + ": " + str(trigrams[i][1]))

    print("\n")
    # lyricsString = ' '.join(lyricsList)
    # fdist = FreqDist(lyricsString)
    # print(name + ": ")
    # print(fdist.most_common(10))

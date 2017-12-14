import csv
import os

from collections import defaultdict

BASE_DIR = ''
GENRE_DIR = os.path.join(BASE_DIR, "singlegenre")

if not os.path.exists(GENRE_DIR):
    os.makedirs(GENRE_DIR)

mixedGenreDict = defaultdict(int)
singleGenreDict = defaultdict(int)

# totalSongs = 0
# multipleGenres = 0
with open('trainingsongdata.csv') as training:
    reader = csv.DictReader(training)
    genres = reader.fieldnames[3:]
    for row in reader:
        mixedGenre = ""
        numGenres = 0
        for genre in genres:
            if row[genre] == "1":
                mixedGenre += genre + " "
                numGenres += 1
        mixedGenreDict[mixedGenre] += 1
        if numGenres == 1 and mixedGenre != "german ":
            singleGenreDict[mixedGenre] += 1
            SINGLE_DIR = os.path.join(GENRE_DIR, mixedGenre[:-1])
            if not os.path.exists(SINGLE_DIR):
                os.makedirs(SINGLE_DIR)
            newSong = open(os.path.join(SINGLE_DIR, str(singleGenreDict[mixedGenre])), "w+")
            newSong.write(row['lyrics'])


# print(len(mixedGenreDict))

# for mixed in mixedGenreDict:
#     print(mixed, mixedGenreDict[mixed])

# print(len(genres))

# print(len(singleGenreDict))

# totalSongs = 0
# for single in singleGenreDict:
#     print(single, singleGenreDict[single])
#     totalSongs += singleGenreDict[single]

# print(totalSongs)




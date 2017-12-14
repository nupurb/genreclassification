import csv

from collections import defaultdict

artistMBTags = defaultdict(set)
artistGenreDict = defaultdict(set)
songdataDict = defaultdict(set)
genreCount = defaultdict(int)

try:
    with open("artist_mbtag_lower.csv", encoding="utf8") as mbtags:
        reader = csv.DictReader(mbtags)
        for row in reader:
            artistMBTags[row["artist_id"]].add(row["mbtag"])
except ValueError:
    print(row, artistMBTags[row["artist_id"]])


with open("artistToIDLower.csv", encoding="utf8") as artistToID:
    reader = csv.DictReader(artistToID)
    for row in reader:
        if len(artistMBTags[row["artist id"]]) != 0:
            artistGenreDict[row["artist name"]] = artistMBTags[row["artist id"]]


with open('songdatalower.csv', encoding='utf8') as songdata:
    reader = csv.DictReader(songdata)
    for row in reader:
        songdataDict[row['artist']].add(row['song'])


print("total artists in genre set", len(artistGenreDict), "\ntotal artists in lyrics set", len(songdataDict))

artistCount = 0
songCount = 0
totalSongCount = 0
for key in songdataDict.keys():
    totalSongCount +=  len(songdataDict[key])
    if key in artistGenreDict.keys():
        songCount += len(songdataDict[key])
        artistCount += 1

print("Artists in both", artistCount, "\nsongs from artists in genre set", songCount, "\ntotal songs in the lyric set", totalSongCount)

for key in artistGenreDict.keys():
    for word in artistGenreDict[key]:
        genreCount[word] += 1

print("number of genres", len(genreCount.keys()))
goodWords = set()
goodWordsList = list()
for word in genreCount.keys():
    if genreCount[word] > 150:
        goodWords.add(word)
        goodWordsList.append(word)

goodSongs = 0

for artist in artistGenreDict:
    for genre in artistGenreDict[artist]:
        if genre in goodWords:
            goodSongs += len(songdataDict[artist])
            break

print("Songs from artists described by top genres", goodSongs)
print(goodWordsList)


with open("trainingsongdata.csv", "w", encoding = "utf8") as training:
    with open("testsongdata.csv", "w", encoding = "utf8") as test:
        fieldnames = ['artist', 'song', 'lyrics'] + goodWordsList
        trainingwriter = csv.DictWriter(training, lineterminator='\n', fieldnames=fieldnames, restval=0)
        testwriter = csv.DictWriter(test, lineterminator='\n', fieldnames=fieldnames, restval=0)
        trainingwriter.writeheader()
        testwriter.writeheader()
        with open('songdatalower.csv', encoding='utf8') as songdata:
            reader = csv.DictReader(songdata)
            trainingCounter = 0
            for row in reader:
                canUse = False
                for genre in artistGenreDict[row['artist']]:
                    if genre in goodWords:
                        canUse = True
                        break
                if canUse:
                    trainingCounter += 1
                    songDict = {'artist': row['artist'], 'song': row['song'], 'lyrics': row['text']}
                    for genre in artistGenreDict[row['artist']]:
                        if genre in goodWords:
                            songDict[genre] = 1
                    if trainingCounter % 20 == 0:
                        testwriter.writerow(songDict)
                    else:
                        trainingwriter.writerow(songDict)




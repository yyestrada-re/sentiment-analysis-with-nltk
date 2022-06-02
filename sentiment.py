import os
import nltk
import string
import numpy as np

from keys import *
from lyricsgenius import Genius
from nltk.sentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer

g_token = Genius(GENIUS_TOKEN)

'''
Returns: The filtered lyrics of the currently playing track on Spotify
Parameters: Current song title and artist name as String parameters
'''
def getLyrics(curr_track, spot_artist):
    # genius data
    raw_lyrics = g_token.search_song(curr_track, spot_artist).to_text()
    print(f"\n-------------------------\nRAW LYRICS:\n{raw_lyrics}\n-------------------------")

    # removing noise from data
    raw_lyrics = raw_lyrics[raw_lyrics.index('Lyrics') + len('Lyrics'):]
    raw_lyrics = raw_lyrics[:raw_lyrics.index('Embed')]

    for word in raw_lyrics:
        if(word == "["):
            nextOccurence = raw_lyrics.index("]")
            raw_lyrics = raw_lyrics[:raw_lyrics.index(word)] + raw_lyrics[nextOccurence + 1:]

    if(raw_lyrics[len(raw_lyrics) - 2].isnumeric()): # second to last
        raw_lyrics = raw_lyrics[:len(raw_lyrics) - 2]
    if(raw_lyrics[len(raw_lyrics) - 1].isnumeric()): # last
        raw_lyrics = raw_lyrics[:len(raw_lyrics) - 1]

    print(f"\nFORMATTED LYRICS:\n{raw_lyrics}\n")
    return raw_lyrics


'''
Returns: A tokenized list of words without stopwords (unimportant words)
Parameters: Untokenized lyrics as a String parameter
'''
def filterTokens(raw_lyrics):
    # tokenize the data --> split the string into a list of tokens
    tokens = nltk.word_tokenize(raw_lyrics)
    #tokens = raw_lyrics.split()
    print(f"RAW TOKENS:\n{tokens}\n")

    # fix abbreviations before removing stopwords
    for word in tokens:
        if word == "ca":
            tokens[tokens.index(word)] = "cannot"
        if word == "'cause":
            tokens[tokens.index(word)] = "because"
        if word == "gonna":
            tokens[tokens.index(word)] = "going to"
        if word == "wanna":
            tokens[tokens.index(word)] = "want to"
        elif "'" in word:
            if len(word) == 2:
                if word[1] == "s":
                    tokens[tokens.index(word)] = "is"
                if word[1] == "d":
                    tokens[tokens.index(word)] = "would"
                if word[1] == "m":
                    tokens[tokens.index(word)] = "am"
            elif len(word) == 3:
                if word[0] == "n" and word[2] == "t":
                    tokens[tokens.index(word)] = "not"
                if word[1] == "r" and word[2] == "e":
                    tokens[tokens.index(word)] = "are"
                if word[1] == "v" and word[2] == "e":
                    tokens[tokens.index(word)] = "have"
                if word[1] == "l" and word[2] == "l":
                    tokens[tokens.index(word)] = "will"

    # remove stopwords from lyrics (arbitrary words)
    stop_words = set(stopwords.words('english'))
    filtered_sentence: list[str] = [word for word in tokens if word.lower() not in stop_words] # filter out words that are in the stopwords db
    print(f"\nSENTENCE W/O STOPWORDS:\n{filtered_sentence}\n")

    # remove punctuation
    for word in filtered_sentence:
        if word[0] in string.punctuation:
            filtered_sentence.remove(word)

    print(f"\nFILTERED SENTENCE:\n{filtered_sentence}\n")
    return filtered_sentence


'''
Returns: The 5 most common words in the song lyrics
Parameters: Tokenized and filtered lyrics
'''
def mostCommon(filtered_lyrics):
    fd = nltk.FreqDist(filtered_lyrics)
    fd.tabulate(5)


'''
Returns: A sentiment score as a float
Parameters: Tokenized and filtered song lyrics
'''
def getSentiment(filtered_sentence):
    # tag these tokens according to what parts of speech they might be (noun, verb, etc):
    tokens = nltk.pos_tag(filtered_sentence)
    print(f"\nTOKENS:\n{tokens}\n")

    sum_scores = []
    pos_scores = []
    neg_scores = []
    compound_scores = []

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    senti = SentimentIntensityAnalyzer()

    for token, tag in tokens:
        if tag.startswith('R'):
            swn_pos = wn.ADV
        elif tag.startswith('V'):
            swn_pos = wn.VERB
        elif tag.startswith('N'):
            swn_pos = wn.NOUN
        else: #tag.startswith('J')
            swn_pos = wn.ADJ

        token = lemmatizer.lemmatize(token, swn_pos) # returns the base word (ex: apples --> apple, builds --> building); groups together different forms of a word to simplify the analyzing process

        # SENTIMENT ANALYSIS WITH VADER --> pre-trained sentiment analizer best suited for language used in social media
        temp_score = senti.polarity_scores(token)
        print(f"VADER POLARITY SCORE FOR \"{token}\": {temp_score}")

        temp_score_neg = senti.polarity_scores(token)['neg']
        temp_score_pos = senti.polarity_scores(token)['pos']
        temp_score_compound = senti.polarity_scores(token)['compound']

        if temp_score_compound != 0:
            compound_scores.append(temp_score_compound)

        # SENTIMENT ANALYSIS WITH SYNSETS
        synsets = list(swn.senti_synsets(token.lower(), pos = swn_pos)) # groupings of synonymous words that express the same concept (nouns, verbs, adjectives, and adverb)
        for synset in synsets[:1]:
            pos_score = synset.pos_score()
            neg_score = synset.neg_score()
            pos_scores.append(pos_score)
            neg_scores.append(neg_score)
            dif = pos_score - neg_score
            if dif != 0:
                sum_scores.append(dif)
            print(f"SYNSET POLARITY SCORE FOR \"{token}\": ['pos': {synset.pos_score()}, 'neg': {synset.neg_score()}, 'obj': {synset.obj_score()}]\n")

    print(f"VADER AVERAGE: {np.average(compound_scores)}")
    print(f"SYNSETS AVERAGE: {np.average(sum_scores)}")

    return np.average(pos_scores) , np.average(neg_scores)

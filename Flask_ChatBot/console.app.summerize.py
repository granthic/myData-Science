
from flask import Flask, render_template, request
#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
### READ THE CSV OR OTHER TYPE OF FILES 
def read_helper_files():
    
    f=open('./data/ecm_text.1.txt','r',errors = 'ignore')
    #f=open('chatbot.mario.txt','r',errors = 'ignore')
    #f=open('chatbot.CCIM.txt','r',errors = 'ignore')
    raw=f.read()
    #raw=f.read()
    raw=raw.lower()# converts to lowercase
    #print(raw)
    return raw

## 
def get_response(userText):
    print("in Get Response")


    # If you get an error uncomment this line and download the necessary libraries
    #nltk.download()

    text = read_helper_files()

    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue

        word = stemmer.stem(word)

        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq



    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))
    print(average)

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    print("Summary {}".format(summary))
    return "get response I am Groot " + userText

def get_bot_response():
    #userText = request.args.get('msg')
    #get_respose(userText)
    #return str(english_bot.get_response(userText))
    #return "I am Groot response " + userText
    userText = "hello"
    return  get_response(userText)
if __name__ == "__main__":
    get_bot_response()

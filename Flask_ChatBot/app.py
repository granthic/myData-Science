""""
ECM Helper bot with Flask, future todo add SF and LOFL helpers
Using summary NLTK 
"""



from flask import Flask, render_template, request,session
from flask_session import Session

#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer

import sys
sys.path.insert(1, r'C:\Users\pmarathe\Documents\Healthfirst\Project\AutoQA\SalesforceDataSync')
from commonUtil.APIClient import APIHelperClass
from commonUtil.dbHelper import DBHelperClass

#nltk.download('popular', quiet=True) # for downloading packages
#nltk.download('punkt') # first-time use only
#   nltk.download('wordnet') # first-time use only

app = Flask(__name__)
app.debug = True
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super_secret_key'
ses = Session()

#session['is_first_question'] ="True"


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","how are you",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello","तू काय म्हणतोस","एक मिनिट थांब", "ठिक आहे","I am glad! You are talking to me"]
    # Preprocessing
lemmer = WordNetLemmatizer()

information_data= ""
#english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
#trainer = ChatterBotCorpusTrainer(english_bot)
#trainer.train("chatterbot.corpus.english")

### READ THE CSV OR OTHER TYPE OF FILES 
def read_helper_files():
    for key in session.keys():
        print(key)
        print(session[key])
    #with open('./data/ecm_text.1.txt','r', encoding='utf8', errors ='ignore') as fin:
    with open('./data/cq_text.txt','r', encoding='utf8', errors ='ignore') as fin:        
        raw = fin.read().lower()
    session['is_first_question'] =True
    print("@@@@ @@@ @@@ Reading FIle {}".format(session['is_first_question'] ))
    information_data= raw
    return raw

#####
## 
def get_response(userText):
    stopWords = set(stopwords.words("english"))
    global information_data
    user = session['is_first_question']
    print("@@@@ @@@ @@@ Is this first Question {}".format(session['is_first_question']))
    if session['is_first_question'] =="False":  
        raw =read_helper_files()

        print("raw {}".format("READ THE FILE>> "))
        session["raw_data"]=raw
    else:
        raw = information_data 
        print(" ** Information Data ==> {}".format("information_data"))
        raw=session["raw_data"]

    #print(raw)
    #TOkenisation
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words

    all_words= nltk.FreqDist(word_tokens)
    #print(all_words.most_common(15))
    
    
    user_response =userText
    print("userText = {}".format(user_response))
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    #vals = cosine_similarity(tfidf[-1], tfidf) #
    vals = linear_kernel(tfidf[-1], tfidf)
    
    idx=vals.argsort()[0][-2] ##   coz most smilar score is user response.. so -2 next most similar 
    #print(vals.argsort())
    #print(tfidf)
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    print(flat)
    all_more_than_1=""
    i=-1
    for flat_i in flat:
        i +=1
        if flat_i>0.3:
            print("flat {} :: {}".format(flat_i, sent_tokens[i]))
            #if flat_i>0.25: ## needs some data relavance so we can 
            all_more_than_1 += sent_tokens[i] + " \n"

            
    print(all_more_than_1)
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        #return robo_response
    else:
        #sent_tokens.remove(user_response)
        #robo_response = summerize_text(all_more_than_1)
        robo_response += robo_response+sent_tokens[idx]
        sent_tokens.remove(user_response)
        #return robo_response

    return "Groot: " + robo_response

#### BOT STUFF 
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    #print("LemNormalize for {}".format(text))
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():   
        if word.lower() in GREETING_INPUTS:
            #session['is_first_question'] ="False"
            return random.choice(GREETING_RESPONSES)

@app.route("/star")
def insert_star_response():
    print("insert_star_response")
    botResponse = request.args.get('star_response')
    print(botResponse)
    response="got star"
    return response

@app.route("/")
def home():
    #session['is_first_question'] = "True"
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    user_response=userText
    response=""

    print(" *** ** * SESSION {}".format(session['is_first_question']))
    #if session['is_first_question'] != "False":
    #    session['is_first_question'] = "True"
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            response="Groot: You are welcome.."
        else:
            if(greeting(user_response)!=None):
                print("Groot: "+greeting(user_response))
                response = greeting(user_response)
                #get_response(userText.lower())
            else:
                #print("ROBO: ",end="")
                response = get_response(userText.lower())
                #print(response(user_response))
    #session['is_first_question'] = "False"
    print(" *** *2* * SESSION {}".format(session['is_first_question']))
    return  response

###### 

def summerize_text(input_text):
    # If you get an error uncomment this line and download the necessary libraries
    #nltk.download()

    text = input_text

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
    #print(average)

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    print("Summary {}".format(summary))
    return summary
####### 





if __name__ == "__main__":
    app.run(debug=True)

import pickle
from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud,STOPWORDS
import pandas as pd
import scattertext as st
import spacy
from pprint import pprint

master = pickle.load( open( "week_sync.p", "rb" ) )
no = 0

rtv_dict = master[0]
tn_dict = master[1]

def pop_hstgs():
    rahul = []
    modi = []
    mega_hstgs = []
    for i in range(40, 52):
        # week = rtv_dict[i]
        # week_hstgs = []
        # for tweet in week:
        #     text = str(tweet.text)
        #     text = re.sub(r'(pic.twitter.com/[A-Za-z0-9]+)','',text)
        #     hstgs = re.findall(r"#(\w+)", text)
        #     week_hstgs.append(hstgs)
        #     flat_list2 = [item for sublist in week_hstgs for item in sublist]
        #     mega_hstgs.append(flat_list2)
            
        # # print(counter)
        # flat_list = list(set(flat_list))
        # r = 0
        # m = 0
        # for el in flat_list:
        #     if "Cong" in el:
        #         r += 1
        #     elif "BJP" in el:
        #         m += 1
        #         print(el)
                
        # print("Rahul:", r, "| Modi", m)
        # rahul.append(r)
        # modi.append(m)
        
        
        
        week = tn_dict[i]
        week_hstgs = []
        for tweet in week:
            text = str(tweet.text)
            text = re.sub(r'(pic.twitter.com/[A-Za-z0-9]+)','',text)
            hstgs = re.findall(r"#(\w+)", text)
            week_hstgs.append(hstgs)
        flat_list2 = [item for sublist in week_hstgs for item in sublist]
        mega_hstgs.append(flat_list2)
        # counter = Counter(flat_list)
        # print(counter.most_common(5))
        
        # flat_list = list(set(flat_list))
        # r = 0
        # m = 0
        # for el in flat_list:
        #     if "RaGa" in el or "Rahul" in el:
        #         r += 1
        #     elif "NaMo" in el or "Modi" in el:
        #         m += 1
        # print("Rahul:", r, "| Modi", m)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    flat_list = [item for sublist in mega_hstgs for item in sublist]
    counter = Counter(flat_list)
    word_cloud = WordCloud(width=1920, height = 1080).generate_from_frequencies(counter)

    plt.imshow(word_cloud)
    plt.axis("off")

    plt.savefig('tn_wordcloud.png', bbox_inches='tight', dpi= 1000)
    
    # x1 = list(np.array(np.linspace(40, 51, 12))+0.15)
    # x2 = list(np.array(np.linspace(40, 51, 12))-0.15)
    # plt.bar(left=x1, width = 0.3, height = rahul, label = 'Rahul Gandhi')
    # plt.bar(left=x2, width = 0.3, height = modi, label = 'Narendra Modi')
    # plt.xticks(range(40,52))
    # plt.ylabel('Mentions in hastags')
    # plt.xlabel('Week Number')
    # plt.title("RepublicTV")
    # plt.legend()
    # plt.show()
    #plt.savefig('rtv_hstgs.png', bbox_inches='tight', dpi= 1000)

def cleanse(d):
    for h in range(40, 52):
        for i in range(len(d[h])):
            text = re.sub(r'(http?:\/\/.*[\r\n]*)|(pic.twitter.com/[A-Za-z0-9]+)',
              '', str(d[h][i].text))
            text = re.sub(r"#(\w+)|(\|)|(-)",'',text)
            text = re.sub(r"(^:)",'',text)
            print(text.strip()+ ":::")

def clean_string(s):
            text = re.sub(r'(http?:\/\/.*[\r\n]*)|(pic.twitter.com/[A-Za-z0-9]+)','', s)
            text = re.sub(r"#(\w+)|(\|)|(-)",'',text)
            text = re.sub(r"(^:)",'',text)
            return text.strip()
    

def likey_tweetey(d , s):
    rts = []
    avg_rt = 0
    avg_likes = 0
    likes = []
    
    for i in range(40, 52):
        for j in range(len(d[i])):
            rts.append([int(d[i][j].retweets) , d[i][j] ])
            avg_likes += int(d[i][j].likes)
            avg_rt += int(d[i][j].retweets)
            likes.append([int(d[i][j].likes) , d[i][j] ])
            
    
    # print("Average likes    : ", avg_likes/no, "Total likes :", avg_likes)
    # print("Average retweets : ", avg_rt/no, "Total retweets :", avg_rt )
    # print("Tweets analyzed  :", no)
    trsort = sorted(rts, reverse = True)
    lsort = sorted(likes, reverse = True)
    # plt.plot([i[0] for i in trsort])
    # print(lsort[0])
    # # plt.yticks(range(0, 1800, 400))
    # plt.title("Retweets per tweet for all tweets sorted - "+s)
    # plt.xlabel("Retweets")
    # plt.show()
    senti_r = 0
    senti_R = 0
    senti_l = 0
    senti_L = 0
    no = 0
    analyser = SentimentIntensityAnalyzer()
    for s, x in zip(trsort, lsort):
        st = str(s[1].text)
        if "cong" in st.lower() or "rahul" in st.lower(): 
            senti_r += TextBlob(clean_string(s[1].text)).sentiment[0]
            senti_R += analyser.polarity_scores(clean_string(s[1].text))['compound']
            senti_l += TextBlob(clean_string(x[1].text)).sentiment[0]
            senti_L += analyser.polarity_scores(clean_string(x[1].text))['compound']
            no+=1
        # print(TextBlob(clean_string(s[1].text)).sentiment[0],"-" ,analyser.polarity_scores(clean_string(s[1].text))['compound'])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Likes" , senti_l/no, senti_L/no)
    print("RTs" , senti_r/no, senti_r/no)

def scatter_skill():
    pass

likey_tweetey(rtv_dict, "RepublicTV")
likey_tweetey(tn_dict, "Times Now")

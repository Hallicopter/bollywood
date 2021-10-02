print("Loading dependencies...")
import pickle
from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
# from wordcloud import WordCloud,STOPWORDS
import pandas as pd
# import scattertext as st
# import spacy
import datetime
from pprint import pprint
from nltk import word_tokenize
from nltk.util import ngrams
import seaborn as sns
print("Loaded dependencies...")
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
    mega_hstgs = [item for sublist in mega_hstgs for item in sublist]
    counter = Counter(mega_hstgs)
    print(counter.most_common(50))
        
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
    
    # flat_list = [item for sublist in mega_hstgs for item in sublist]
    # counter = Counter(flat_list)
    # word_cloud = WordCloud(width=1920, height = 1080).generate_from_frequencies(counter)

    # plt.imshow(word_cloud)
    # plt.axis("off")

    # plt.savefig('tn_wordcloud.png', bbox_inches='tight', dpi= 1000)
    
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
    
def likey_tweetey(d, d2 , s):
    rts = []
    rts2 = []
    avg_rt = 0
    avg_likes = 0
    likes = []
    likes2= []
    
    for i in range(40, 52):
        for j in range(len(d[i])):
            rts.append([int(d[i][j].retweets) , d[i][j] ])
            avg_likes += int(d[i][j].likes)
            avg_rt += int(d[i][j].retweets)
            likes.append([int(d[i][j].likes) , d[i][j] ])
    
    for i in range(40, 52):
        for j in range(len(d2[i])):
            rts2.append([int(d2[i][j].retweets) , d2[i][j] ])
            avg_likes += int(d2[i][j].likes)
            avg_rt += int(d2[i][j].retweets)
            likes2.append([int(d2[i][j].likes) , d2[i][j] ])
            
            
    
    # print("Average likes    : ", avg_likes/no, "Total likes :", avg_likes)
    # print("Average retweets : ", avg_rt/no, "Total retweets :", avg_rt )
    # print("Tweets analyzed  :", no)
    trsort = sorted(rts, reverse = True)
    trsort2 = sorted(rts2, reverse = True)
    lsort = sorted(likes, reverse = True)
    lsort2 = sorted(likes2, reverse = True)

    print([(el[1].text,el[0]) for el in trsort2[:5]])
    # y = [i[0] for i in trsort][:5]
    # y2 = [i[0] for i in trsort2][:5]
    # x1 = [-5]*5
    # x2 = [5]*5
    # fig, ax = plt.subplots()
    # ax.scatter(x1, y)
    # ax.scatter(x2, y2)
    # # plt.yticks(range(50000,320000,20000))
    # n=[str(i[1].text) for i in trsort][:5]
    # n2=[str(i[1].text) for i in trsort2][:5]
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # for i, txt in enumerate(n):
    #     ax.annotate(txt, (x1[i],y[i]), color = "white", horizontalalignment='center',
    #      backgroundcolor='blue',wrap=True)
    # for i, txt in enumerate(n2):
    #     ax.annotate(txt, (x2[i],y2[i]), color = "white", horizontalalignment='center',
    #                 backgroundcolor='red',wrap=True)
    # plt.xticks(range(-10,10))
    # plt.show()
    # plt.plot([i[0] for i in trsort])
    # print(lsort[0])
    # # plt.yticks(range(0, 1800, 400))
    # plt.title("Retweets per tweet for all tweets sorted - "+s)
    # plt.xlabel("Retweets")
    # plt.show()
    # senti_r = 0
    # senti_R = 0
    # senti_l = 0
    # senti_L = 0
    # no = 0
    # analyser = SentimentIntensityAnalyzer()
    # for s, x in zip(trsort, lsort):
    #     st = str(s[1].text)
    #     if "bjp" in st.lower() : 
    #         senti_r += TextBlob(clean_string(s[1].text)).sentiment[0]
    #         senti_R += analyser.polarity_scores(clean_string(s[1].text))['compound']
    #         senti_l += TextBlob(clean_string(x[1].text)).sentiment[0]
    #         senti_L += analyser.polarity_scores(clean_string(x[1].text))['compound']
    #         no+=1
    #     # print(TextBlob(clean_string(s[1].text)).sentiment[0],"-" ,analyser.polarity_scores(clean_string(s[1].text))['compound'])
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("Likes VADER" ,senti_L/no, no)
    # print("RTs VADER" , senti_r/no)

def scatter_skill():
    for_df = []
    labels = ["Channel", "Tweet"]
    for i in range(40, 52):
        for j in range(len(rtv_dict[i])):
            s = re.sub(r'(http?:\/\/.*[\r\n]*)|(pic.twitter.com/[A-Za-z0-9]+)','', str(rtv_dict[i][j].text))
            for_df.append(("RepublicTV", s))
    
    for i in range(40, 52):
        for j in range(len(tn_dict[i])):
            s = re.sub(r'(http?:\/\/.*[\r\n]*)|(pic.twitter.com/[A-Za-z0-9]+)','', str(tn_dict[i][j].text))
            for_df.append(("Times Now", s))
    
    df = pd.DataFrame.from_records(for_df, columns=labels)
    nlp = spacy.load('en')
    corpus = st.CorpusFromPandas(df, 
                                category_col='Channel', 
                                text_col='Tweet',
                                nlp=nlp).build()
    html = st.produce_scattertext_explorer(corpus,
                                            category='Times Now',
                                            category_name='Times Now TV',
                                            not_category_name='RepublicTV News',
                                            width_in_pixels=1000)
    open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))

def n_grams(n, d):
    text = ""
    for i in range(40, 52):
        for j in range(len(d[i])):
            s = re.sub(r'(http?:\/\/.*[\r\n]*)|(pic.twitter.com/[A-Za-z0-9]+)','', str(d[i][j].text))
            s = " " + s
            text += s
    
    # token = nltk.word_tokenize(text)
    stop = set(stopwords.words('english'))
    
    tokenizer = RegexpTokenizer(r'\w+')
    token = tokenizer.tokenize(text)
    filtered_sentence = [w for w in token if not w in stop]

    grams = ngrams(filtered_sentence, n)
    dict1 = list(Counter(grams).most_common(30))
    dict2 = list(dict1)
    dat = list([int(s[1]) for s in dict1])
    lab = list([" ".join(s[0]) for s in dict2])
    # plt.plot(Counter(grams).most_common(20))
    print(lab)
    x = np.arange(len(dat))
    plt.bar(x, dat)
    for a,b,c in zip(x, lab, dat):
        plt.text(a, c, b, rotation = 'vertical',
                verticalalignment='top', horizontalalignment='center', color='white')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='off')

    plt.title("Most common words in a tweet.")
    plt.show()

def heatmap(d1, s):
    freq_matrix = np.zeros((12, 7))
    for i in range(40, 52):
        for j in range(len(d1[i])):
            d1[i][j].timestamp += datetime.timedelta(hours=5.5)
            freq_matrix[i%40,(d1[i][j].timestamp.isocalendar()[2]-1)]+=1
    # for i in range(40, 52):
    #     for j in range(len(d2[i])):
    #         d2[i][j].timestamp += datetime.timedelta(hours=5.5)
    #         freq_matrix[i%40,(d2[i][j].timestamp.isocalendar()[2]-1)]+=1
    # print(freq_matrix)
    
    ax = sns.heatmap(freq_matrix, annot=False, fmt=".1f", vmin=0, vmax=430)
    ax.set_xticklabels(["Monday", "Tueday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    ax.set_yticklabels(range(51,39,-1))
    ax.set_ylabel("Week Number")
    ax.set_xlabel("Day of the week")
    ax.set_title("Heatmap of tweet frequency - " + s)
    plt.show()

def get_range(n):
    if 0<=n<=1:
        return 0    
    if 2<=n<=3:
        return 1
    if 4<=n<=5:
        return 2
    if 6<=n<=7:
        return 3
    if 8<=n<=9:
        return 4
    if 10<=n<=11:
        return 5
    if 12<=n<=13:
        return 6
    if 14<=n<=15:
        return 7
    if 16<=n<=17:
        return 8
    if 18<=n<=19:
        return 9
    if 20<=n<=21:
        return 10
    if 22<=n<=23:
        return 11   

def heatmap_day(d, s):
    freq_matrix = np.zeros((7, 12))
    for i in range(40, 52):
        for j in range(len(d[i])):
            # print(d[i][j].timestamp.hour)
            d[i][j].timestamp += datetime.timedelta(hours=5.5)
            
            freq_matrix[(d[i][j].timestamp.isocalendar()[2]-1), get_range(d[i][j].timestamp.hour)]+=1
    
    ax = sns.heatmap(freq_matrix, annot=False, fmt=".1f")
    ax.set_yticklabels(["Monday", "Tueday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][::-1])
    ax.set_xticklabels(["12-2 AM", "2-4 AM", "4-6 AM", "6-8 AM", "8-10 AM", "10-12 AM", "12-2 PM",
                        "2-4 PM", "4-6 PM", "6-8 PM", "8-10 PM", "10-12 AM"])
    ax.set_xlabel("Time of the day")
    ax.set_ylabel("Day of the week")
    ax.set_title("Heatmap of tweet frequency - " + s)
    plt.show()

def sent(d, s):
    analyser = SentimentIntensityAnalyzer()
    senti = 0
    neg = 0
    pos = 0
    neu = 0
    no = 0
    neg_c = 0
    pos_c = 0
    for i in range(40, 52):
        for j in range(len(d[i])):
            st = str(d[i][j].text)
            if s in st.lower():
                net = analyser.polarity_scores(clean_string(st))['compound']
                senti += net
                if net<-0.2:
                    neg_c += 1
                    
                if net>0.2:
                    pos_c += 1
                    
                neg += analyser.polarity_scores(clean_string(st))['neg']
                pos += analyser.polarity_scores(clean_string(st))['pos']
                neu += analyser.polarity_scores(clean_string(st))['neu']
                no+=1
    print("VADER" , senti/no, no, s)
    print("Neutral", neu/no, "Positive", pos/no, "Negative", neg/no)
    print("% negative", neg_c/no * 100)
    print("% positive", pos_c/no * 100)
    print("% neutral", 100 - neg_c/no * 100 - pos_c/no * 100)
    plt.pie([pos_c, neg_c, no - pos_c - neg_c], labels = ["Positive", "Negative", "Neutral"])
    plt.show()
            

likey_tweetey(tn_dict, rtv_dict, "RepublicTV")
# likey_tweetey(tn_dict, "Times Now")
# scatter_skill()

# n_grams(2, rtv_dict)
# n_grams(2, tn_dict)

# heatmap(rtv_dict,"RepublicTV")
# pop_hstgs()

from twitterscraper import query_tweets
from pattern.en import tag
from pattern.en import sentiment
import re
import pickle

no = 0
master = []

for tweet in query_tweets("from%3Arepublic&src=typd", 20000):
   
    no += 1 
    # print(tweet.text.encode('utf-8'))
    
    hstgs = re.findall(r"#(\w+)", str(tweet.text.encode('utf-8')))
    if hstgs != []:
        print(str(no)+ " : ")
        
        # print(tweet.timestamp)
        hstgs.append(str(tweet.timestamp))
        print(hstgs)
        master.append(tweet)
    # print(sentiment(str(tweet.text.encode('utf-8'))))
    # for word, pos in tag(str(tweet.text.encode('utf-8'))) :
    #     if pos == "JJ": # Retrieve all adjectives.
    #         print ("Tag --> "+ word+"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
pickle.dump( master, open( "arnab_big_time.p", "wb" ) )
import pickle
from collections import Counter

master1 = pickle.load( open( "arnab_big_time.p", "rb" ) )
master2 = pickle.load( open( "times_now_big_time.p", "rb" ) )
# flat_list = [item for sublist in master for item in sublist]

tn_dict = {}
rtv_dict = {}

for i in range(40, 52):
    tmp = []
    for rtv in master1:
        if rtv.timestamp.isocalendar()[1] == i:
            tmp.append(rtv)
    
    rtv_dict[i] = tmp

    tmp = []
    for tn in master2:
        if tn.timestamp.isocalendar()[1] == i:
            tmp.append(tn)
    tn_dict[i] = tmp

master = (rtv_dict, tn_dict)
pickle.dump( master, open( "week_sync.p", "wb" ) )
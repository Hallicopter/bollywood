import pickle
from collections import Counter

master = pickle.load( open( "times_now_big.p", "rb" ) )
flat_list = [item for sublist in master for item in sublist]
print(len(flat_list))
print(flat_list[:10])
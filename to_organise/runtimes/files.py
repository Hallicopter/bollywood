import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode()
%matplotlib notebook

data = json.load(open('betatest.json'))

runtimes = []
releases = []
for datum in data:
    if datum['runtime'] != 'N/A' and int(datum['runtime']) > 30:
        runtimes.append(int(datum['runtime']))
        releases.append(str(datum['release']))
        print(int(datum['runtime']), datum['film_name'][0])
# print(len(runtimes), len(releases))

fig, ax = plt.subplots(1,1) 
ax.plot(runtimes)
ax.set_xticks(range(0, len(releases),99))
ax.set_xticklabels(releases[::99], rotation='vertical', fontsize=9)
sns.set()
sns.set_style("dark")
coefficients, residuals, _, _, _ = np.polyfit(range(len(runtimes)),runtimes,1,full=True)
plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(runtimes))])
plt.xlabel("Year of release")
plt.ylabel("Runtime (in minutes)")
plt.title("Movie runtimes over the 50 years of bollywood")
fig = plt.gcf() # "Get current figure"
py.iplot_mpl(fig)
# plt.show()
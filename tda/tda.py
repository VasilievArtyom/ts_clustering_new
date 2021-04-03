import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path

from ripser import Rips
import persim

plt.rc('text', usetex=True)


MAX_BLC_ID = 3400
embedding_dimension=2

diagramms = []
rips = Rips(maxdim=1, coeff=2)


def takensEmbedding(_data, _delay = 0, _dimension=embedding_dimension):
    embeddedData = np.array([_data[0:len(data)-_delay*_dimension]])
    for i in range(1, _dimension):
        embeddedData = np.append(embeddedData, [_data[i*_delay:len(data) - _delay*(_dimension - i)]], axis=0)
    return np.transpose(np.squeeze(embeddedData))
    

for current_id in range(MAX_BLC_ID + 1):
   data = pd.read_csv('../regularization_and_split/aftersplit/id{:04d}.csv'.format(current_id)).to_numpy()
   current_diagramm = rips.fit_transform(takensEmbedding(data)[:1100, :])
   diagram_h1 = current_diagramm[1]
   print(current_id)
   fig, ax = plt.subplots(figsize=(6, 5))
   rips.plot(current_diagramm, show=False)
   plt.title("PD of $H_k$ for id{:04d}".format(current_id))
   plt.tight_layout()
   plt.draw()
   fig.savefig("diagramms/id{:04d}".format(current_id))
   diagramms.append(current_diagramm)

product = ((i, j) for i in range(MAX_BLC_ID + 1) for j in range(MAX_BLC_ID + 1))
distances = np.zeros((2, (MAX_BLC_ID + 1), (MAX_BLC_ID + 1)))
for i, j in product:
    # distances[0, i, j] = persim.sliced_wasserstein(diagramms[i][0], diagramms[j][0])
    # distances[1, i, j] = persim.sliced_wasserstein(diagramms[i][1], diagramms[j][1])
    distances[0, i, j] = persim.bottleneck(diagramms[i][0], diagramms[j][0])
    distances[1, i, j] = persim.bottleneck(diagramms[i][1], diagramms[j][1])
T_n_labels = ["id{:04d}".format(tmp_i) for tmp_i in range(MAX_BLC_ID + 1)]
pd.DataFrame(distances[0, :, :], columns = T_n_labels, index=T_n_labels).to_csv('distancesH0.csv')
pd.DataFrame(distances[1, :, :], columns = T_n_labels, index=T_n_labels).to_csv('distancesH1.csv')

outpath = ""


fig, ax = plt.subplots(figsize=(60, 60))
im = ax.imshow(distances[0, :, :])
ax.set_xticks(np.arange(len(T_n_labels)))
ax.set_yticks(np.arange(len(T_n_labels)))

ax.set_xticklabels(T_n_labels)
ax.set_yticklabels(T_n_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(MAX_BLC_ID + 1):
#     for j in range(MAX_BLC_ID + 1):
#         text = ax.text(j, i, np.round(distances[0, i, j],1),
#                        ha="center", va="center", color="w", size=8)

# ax.set_title("Pearson correlation coefficient")
fig.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "distances.png"))
plt.clf()
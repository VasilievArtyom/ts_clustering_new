import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


df = pd.read_csv('../autoencoder/encoder_output.csv')
df.head(5)
df.info()


#convert datetime to seconds
from datetime import datetime
def to_total_seconds(s):
    pos = s.find('.')
    #startpoin is 2020-10-20 12:42:31.559564
    if pos > 0:
        return (datetime.fromisoformat(s+'0'*(26-len(s))) - datetime(2020,10,20, 12, 42, 31)).total_seconds() - .559564
    else:
        return (datetime.fromisoformat(s+'.'+'0'*6) - datetime(2020,10,20, 12, 42, 31)).total_seconds() - .559564

df['reg_time'] = df['reg_time'].apply(to_total_seconds)
df['reg_time'].head()


#encode element_name by integer-valued id
unique_names = df['element_name'].unique()
f = open("element_name_ids.csv","w")
name_to_id_dict = {}
for tmp_id, name in enumerate(unique_names):
    name_to_id_dict[name] = tmp_id
    print(name, tmp_id, sep=',', file=f)
f.close()
id_max = len(unique_names) - 1
df['id'] = np.asarray([name_to_id_dict[str(x)] for x in df['element_name']])
df['id'].head()


#split dataframe by id 
df_per_id_list = [df[df['id'] == _id] for _id in range(0, len(unique_names))]
df_per_id_list[0].info()

#find diff for all sequential entries t_i and t_{i-1}
diffs = []
for _id, _df in enumerate(df_per_id_list):
    if len(_df.index) < 1000:
        diffs.append(np.array([0]))
        continue
    df_unshifted = _df.iloc[:-1].reset_index(drop=True)
    df_shifted = _df.iloc[1:].reset_index(drop=True)
    time_diffs = (df_shifted['reg_time'] - df_unshifted['reg_time']).values
    # cols = [col for col in _df.columns if col not in ['reg_time']]
    cols = ['encoded_data']
    features_diffs = np.sqrt((((df_shifted[cols] - df_unshifted[cols])**2).sum(axis = 1)).values)
    features_diffs = (features_diffs-features_diffs.min())/(features_diffs.max()-features_diffs.min())
    tmp = np.sort(np.multiply(time_diffs, features_diffs))
    diffs.append(tmp[(tmp != 0.0) & (tmp < 10)])
    if len(tmp[tmp != 0.0]) == 0:
        raise NameError('Err') 
print(diffs[0])
print(len(diffs))

# collect representative ids with at least 1000 entries
id_list = [x for x in range(0, len(diffs)) if len(diffs[x]) > 1000]
print(len(id_list))

print_id = 42
fig, ax = plt.subplots(figsize=(6, 5))
ax.hist(diffs[id_list[print_id]], bins=1000, density=True)
plt.xlabel('$\Delta t$')
plt.ylabel('$H[\Delta t]$')
# plt.xlim(-0.001, 0.4)
plt.ylim(0, np.max(diffs[id_list[print_id]])*1.1)
plt.title("id = {}".format(id_list[print_id]))
plt.tight_layout()
plt.draw()
# plt.show()
fig.savefig("plots/dstr/id{}.png".format(id_list[print_id]))


print(df['reg_time'])

for _id, _df in enumerate(df_per_id_list):
   _df[['encoded_data']].to_csv('aftersplit/id{:03d}.csv'.format(_id), index=False)
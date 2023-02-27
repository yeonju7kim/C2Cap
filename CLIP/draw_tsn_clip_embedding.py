# -*- encoding: utf-8 -*-
'''
@File    :   draw_tsn_clip_embedding.py   
@Contact :   yeonju7.kim@gmail.com
@License :   (C)Copyright 2022-2024
 
@Modify Time      @Author        @Version    @Desciption
------------      -------        --------    -----------
2023-02-01 오전 10:47   yeonju7.kim      1.0         None
'''
import pickle
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

with open('clip_collections_pkl.pkl', 'rb') as f:
    clip_collections = pickle.load(f)

features = clip_collections.features
labels = clip_collections.labels
centroids = clip_collections.centroids

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(x)

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title)

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title)

import matplotlib.pyplot as plt
plt.show()

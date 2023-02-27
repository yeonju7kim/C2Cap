# -*- encoding: utf-8 -*-
'''
@File    :   extract_clip_feature.py   
@Contact :   yeonju7.kim@gmail.com
@License :   (C)Copyright 2022-2024
 
@Modify Time      @Author        @Version    @Desciption
------------      -------        --------    -----------
2023-01-25 오전 11:42   yeonju7.kim      1.0         None
'''
import argparse
import json
import os
import pickle
from collections import defaultdict

import clip
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from torchvision.transforms import RandAugment
import matplotlib.pyplot as plt

class CLIPFeatureExtractor():
    def __init__(self, clip_feature_hdf, train_folder,valid_folder,annotation_path, aug_num):
        self.clip_feature_hdf = clip_feature_hdf
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.train_annotation_path = annotation_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rand_aug = RandAugment(magnitude=1)
        self.aug_num = aug_num
        if os.path.exists(self.clip_feature_hdf) == False:
            self.save_clip_feature_hdf()

    def _get_clip_feature(self, image):
        if hasattr(self, 'model') == False:
            self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = np.array(self.model.encode_image(image_input).to('cpu'))
        return image_features

    def save_clip_feature_hdf(self):
        h5py_file = h5py.File(self.clip_feature_hdf, 'w')
        with torch.no_grad():
            for idx, file in enumerate(tqdm(os.listdir(self.train_folder))):
                file = os.path.join(self.train_folder, file)
                image = Image.open(file)
                # plt.imshow(self.rand_aug(image))
                # plt.show()
                clip_features = np.array([self._get_clip_feature(self.rand_aug(image))[0] for i in range(self.aug_num)])
                clip_features = np.append(clip_features, np.array(self._get_clip_feature(image)), axis=0)
                img_id = (str)((int)(file.split('_')[-1].split('.')[0]))
                new_group = h5py_file.create_group(img_id)
                new_group.create_dataset('image', data=clip_features)
            for idx, file in enumerate(tqdm(os.listdir(self.valid_folder))):
                file = os.path.join(self.valid_folder, file)
                image = Image.open(file)
                # plt.imshow(self.rand_aug(image))
                # plt.show()
                clip_features = np.array([self._get_clip_feature(self.rand_aug(image))[0] for i in range(self.aug_num)])
                clip_features = np.append(clip_features, np.array(self._get_clip_feature(image)), axis=0)
                img_id = (str)((int)(file.split('_')[-1].split('.')[0]))
                new_group = h5py_file.create_group(img_id)
                new_group.create_dataset('image', data=clip_features)
        h5py_file.close()

    def load_clip_feature_hdf(self):
        h5py_file = h5py.File(self.clip_feature_hdf, 'r')
        image_ids_in_file = list(h5py_file.keys())
        image_features = []
        image_ids = []
        for idx, img_id in enumerate(tqdm(image_ids_in_file)):
            image_list = np.array(h5py_file[img_id]['image'])
            for img in image_list:
                image_features.append(img)
                image_ids.append(img_id)
        return np.array(image_features), np.array(image_ids)

class CLIPFeatureCollection():
    def __init__(self, features, labels, centroids):
        self.features = features
        self.labels = labels
        self.centroids = centroids

class CLIPClusterConstructor():
    def __init__(self, clip_feature_hdf, train_folder, valid_folder, annotation_path, clip_cluster_pkl, clip_collections_pkl, k, aug_num):
        self.clip_cluster_pkl = clip_cluster_pkl
        self.clip_collections_pkl = clip_collections_pkl
        if os.path.exists(self.clip_cluster_pkl) == False:
            clip_feature_train = CLIPFeatureExtractor(clip_feature_hdf, train_folder, valid_folder, annotation_path, aug_num)
            image_features, cluster_ids = clip_feature_train.load_clip_feature_hdf()
            cluster_ids = self._cluster_kmean(image_features, k)
            self.save_clip_cluster(image_features, cluster_ids)

    @staticmethod
    def _cluster_kmean(x, k):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
        return kmeans.labels_

    def load_clip_cluster(self):
        with open(self.clip_cluster_pkl, 'rb') as f:
            clip_feature_cluster = pickle.load(f)
            return np.array([feature for id, feature in clip_feature_cluster.items()]), clip_feature_cluster

    def load_clip_collections(self):
        with open(self.clip_collections_pkl, 'rb') as f:
            clip_collections = pickle.load(f)
            return clip_collections

    def _average_clip_cluster(self, image_features, cluster_ids):
        features_by_cluster = defaultdict(list)
        for (f, l) in zip(image_features, cluster_ids):
            features_by_cluster[l].append(f)
        averaged_features_in_cluster = {}
        for k, v in features_by_cluster.items():
            averaged_features_in_cluster[k] = np.sum(np.array(v), axis=0) / len(v)
        return averaged_features_in_cluster

    def _set_ids_to_cluster(self, image_label, cluster_ids):
        self.image_ids_to_cluster = defaultdict(list)
        for (l, id) in zip(image_label, cluster_ids):
            self.image_ids_to_cluster[id] = l

    def save_clip_cluster(self, image_features, cluster_ids):
        averaged_features_in_cluster = self._average_clip_cluster(image_features, cluster_ids)
        clip_feature_collections = CLIPFeatureCollection(image_features, cluster_ids, averaged_features_in_cluster)
        with open(self.clip_cluster_pkl, 'wb') as f:
            pickle.dump(averaged_features_in_cluster, f)
        with open(self.clip_collections_pkl, 'wb') as f:
            pickle.dump(clip_feature_collections, f)

    def _separate_image_by_cluster(self):
        raise NotImplementedError


class CLIPConfounder():
    def __init__(self, clip_cluster_pkl):
        self.clip_cluster_pkl = clip_cluster_pkl
        self.load_clip_cluster()

    def load_clip_cluster(self):
        with open(self.clip_cluster_pkl, 'rb') as f:
            clip_feature_cluster = pickle.load(f)
            self.confounder_dictionary = np.array([feature for id, feature in clip_feature_cluster.items()])


def arg_parse():
    parser = argparse.ArgumentParser(description='Clip Cluster Confounder')
    parser.add_argument('--clip_feature_hdf', type=str)
    parser.add_argument('--train_folder', type=str, default='D:/data/image_captioning/github_ignore_material/raw_data/MS_COCO_2014/train2014/img')
    parser.add_argument('--valid_folder', type=str, default='D:/data/image_captioning/github_ignore_material/raw_data/MS_COCO_2014/val2014/img')
    parser.add_argument('--train_annotation_path', type=str, default="D:/data/image_captioning/caption_data/annotations_trainval2014/annotations/captions_train2014.json")
    parser.add_argument('--clip_cluster_pkl', type=str)
    parser.add_argument('--clip_collections_pkl', type=str)
    parser.add_argument('--k', type=int, default=1000)
    parser.add_argument('--aug_num', type=int, default=3)
    args = parser.parse_args()
    return args

def normalize(clip_dictionary):
    mean_clip_embedding = clip_dictionary.mean(axis=0)
    normalize_clip_dictionary = []
    for clip_embedding in clip_dictionary:
        normalize_clip_dictionary.append(clip_embedding-mean_clip_embedding)
    return np.array(normalize_clip_dictionary)

if __name__ == '__main__':
    args = arg_parse()

    if args.clip_feature_hdf == None:
        args.clip_feature_hdf = 'clip_feature_train.hdf5'
        args.clip_feature_hdf = f'{args.clip_feature_hdf.split(".")[0]}_aug{args.aug_num}_all_image_mag1.{args.clip_feature_hdf.split(".")[1]}'
    if args.clip_cluster_pkl == None:
        args.clip_cluster_pkl = 'clip_feature_train_cluster.pkl'
        args.clip_cluster_pkl = f'{args.clip_cluster_pkl.split(".")[0]}_k{args.k}_aug{args.aug_num}_all_image_mag1.{args.clip_cluster_pkl.split(".")[1]}'
    if args.clip_collections_pkl == None:
        args.clip_collections_pkl = 'clip_collections.pkl'
        args.clip_collections_pkl = f'{args.clip_collections_pkl.split(".")[0]}_k{args.k}_aug{args.aug_num}_all_image_mag1.{args.clip_collections_pkl.split(".")[1]}'

    clip_cluster = CLIPClusterConstructor(args.clip_feature_hdf, args.train_folder, args.valid_folder, args.train_annotation_path, args.clip_cluster_pkl, args.clip_collections_pkl, args.k, args.aug_num)
    clip_cluster_list, clip_cluster_dictionary = clip_cluster.load_clip_cluster()
    clip_collections = clip_cluster.load_clip_collections()
    # normalized_clip_cluster_list = normalize(clip_cluster_list)
    # clip_cluster_pkl_normalized = f'{args.clip_cluster_pkl.split(".")[0]}_normalized.{args.clip_cluster_pkl.split(".")[1]}'
    # with open(clip_cluster_pkl_normalized, 'wb') as f:
    #     pickle.dump(normalized_clip_cluster_list, f)
    clip_confounder = CLIPConfounder(args.clip_cluster_pkl)
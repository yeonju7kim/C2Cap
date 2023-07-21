# C2 dictionary
CLIP-Confounder-free Dictionary for Image Captioning

## How to get the C2 dictionary

Run `create_clip_feature_dictionary.py` using the following arguments: 

| Argument | Possible values                                  |
|------|--------------------------------------------------|
| `--clip_feature_hdf` | filename for clip embedding for training dataset |
| `--train_folder` | path for coco train dataset                      |
| `--valid_folder` | path for coco validation dataset                 |
| `--clip_cluster_pkl` | filename for clip cluster(confounder dictionary) |
| `--clip_collections_pkl` | filename for clip_collection                     |
| `--k` | number of clusters for kmeans(default: 1000)     |
| `--aug_num` | number of augmented pictures(default: 3)         |
| `--aug_mag` | degree of random augmentation(default: 1)        |


## Output

- clip_feature_train.hdf5: All CLIP feature of the training data
- clip_feature_train_cluster.pkl: centroids of K-means clusters, confounder dictionary
- clip_collections.pkl: Object which contains all the information about the CLIP feature dictionary(image_features, cluster_ids, averaged_features_in_cluster)
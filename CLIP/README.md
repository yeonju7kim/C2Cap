# C3 dict.
CLIP Contextual Confounder Dictionary for Image Captioning

## How to get the C3 dict

Run `clip_feature_dictionary.py` using the following arguments: 

| Argument | Possible values |
|------|-----------------|
| `--clip_feature_hdf` |                 |
| `--train_folder` |                 |
| `--train_annotation_path` |                 |
| `--clip_cluster_pkl` |                 |
| `--k` | (default: 1000) |

## Output

- clip_feature_train.hdf5: All CLIP feature of the training data
- clip_feature_train_cluster.pkl: K-means cluster의 centroid
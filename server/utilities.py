import numpy as np
import pickle
import random
import vptree as vp


def get_euclidean(x, y):
  euclidean_distance = np.sqrt(np.sum(np.power(x - y, 2)))
  return euclidean_distance


def normalize_embedding(x):
  return x / np.linalg.norm(x, axis=1, ord=1, keepdims=True)


def preprocess_request(vector):
  normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
  reshaped_vector = np.reshape(normalized_vector, (1, vector.shape[0], 1))
  return reshaped_vector


def construct_tree(points, dist_fn, tree_path):
  print("constructing VP Tree")
  tree = vp.VPTree(points, dist_fn)

  print("serializing VP Tree")
  with open(tree_path, 'wb') as f:
    f.write(pickle.dumps(tree))


def read_database(tree_path, index_dict_path):
  if tree_path == None:
    index_dict = pickle.loads(open(index_dict_path, 'rb').read())
    return index_dict
  if index_dict_path == None:
    tree = pickle.loads(open(tree_path, 'rb').read())
    return tree
  index_dict = pickle.loads(open(index_dict_path, 'rb').read())
  tree = pickle.loads(open(tree_path, 'rb').read())
  return (tree, index_dict)


def search_embedding(query_embedding, tree, index_dict, limit):
  nearest_embeddings = []
  results = tree.get_n_nearest_neighbors(query_embedding, limit)
  results.sort(key=lambda e: e[0])
  return results

#   for i,j in result:
#     path = index_dict[tuple(j)]
#     title = path.split('_')[1].split('.')[0]
#     eu_dist = i


#     nearest_images.append({
#       "path": path,
#       "title": title,
#       "eu_dist": eu_dist
#     })
#   return nearest_images


def np2csv(x, file_name):
    np.savetxt(file_name, x, delimiter=",", header="data")
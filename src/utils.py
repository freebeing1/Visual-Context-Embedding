import numpy as np
import os
from glob import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def euclidean_dis(A, B):
    return np.linalg.norm(np.asarray(A)-np.asarray(B))


def perceptual_dis(A, B):
    x_dis = np.abs(np.subtract(A.x, B.x))
    y_dis = np.abs(np.subtract(A.y, B.y))

    w_avg = np.average(np.array([A.w, B.w]))
    h_avg = np.average(np.array([A.h, B.h]))

    if euclidean_dis(A.center, B.center) <= 0:
        return 0
    elif (w_avg * x_dis + h_avg * y_dis) <= 0:
        return 0
    else:
        return euclidean_dis(A.center, B.center) * (x_dis + y_dis) / (w_avg * x_dis + h_avg * y_dis)


def check_file_in_dir(check_dir, check_file=None):
    if check_file is not None:
        f_name = os.path.join(check_dir, check_file)
        if os.path.isfile(f_name):
            print('"{}" already exists.\n'.format(f_name))
            return True
    elif glob(check_dir+'*'):
        print('Some files already exist in "{}".\n'.format(check_dir))
        return True
    else:
        return False


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        return d
    return d


def min_max_normalize(A):
    max_val = np.array(A).max()
    min_val = np.array(A).min()
    return list([(val-min_val)/(max_val-min_val) for val in A])


def l2_normalize_embedding(target_vector, dim=32):
    if target_vector.shape[0] is not dim:
        arranged = np.transpose(target_vector)
    else:
        arranged = target_vector

    # print(transposed.shape) # (dim, n_class)
    n_class = arranged.shape[1]

    normalized = np.empty(arranged.shape)

    for i in range(n_class):
        norm = np.linalg.norm(arranged[:, i])
        normalized[:, i] = arranged[:, i] / norm

    return normalized


def visualize_vector(embedding_dir, save_dir='../result/fig/'):

    _, vce_name = os.path.split(embedding_dir)

    tsne = TSNE(n_components=2).fit_transform(np.transpose(np.load(embedding_dir)))
    _, ax = plt.subplots()
    ax.scatter(tsne[:, 0], tsne[:, 1], s=1)
    
    plt.savefig(os.path.join(save_dir, vce_name.split('.')[0]+'.png'))
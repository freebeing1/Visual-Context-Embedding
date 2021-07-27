import json
import numpy as np
import os

from utils import check_file_in_dir, make_dir, perceptual_dis, l2_normalize_embedding, visualize_vector
from models import ImageJSON, Synset, Vocabulary, CoOccurrenceMatrix
from glob import glob
from glove import Glove


def json_preprocessing(json_file='../data/vg-metadata/objects.json',
                       train_dir='../data/vg-metadata/by-id-train/',
                       test_dir='../data/vg-metadata/by-id-test/',
                       test_list_file='../data/vg-metadata/vg_test_list.txt'):
    print('Preprocess json file : [{}]\n'.format(json_file))
    make_dir(train_dir)
    make_dir(test_dir)

    if check_file_in_dir(train_dir):return -1

    with open(test_list_file, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        test_list = list([int(x.replace('\n', '')) for x in lines])

    json_data = json.load(open(json_file, 'r'))
    for image in json_data:
        img_json = ImageJSON(image_id=image['image_id'])
        img_json.init_objects(image['objects'])
        if image['image_id'] in test_list:
            img_json.save(path=test_dir)
        else:
            img_json.save(path=train_dir)


def initialize_vocab(min_tf=10,
                     metadata_dir='../data/vg-metadata/by-id-train/',
                     vocab_dir='../result/vocab/'):
    print('Initialize vocab with minimum word count [{}]\n'.format(min))

    if check_file_in_dir(check_dir=vocab_dir, check_file='vocab.txt'): return -1

    v = Vocabulary()
    vocab = Vocabulary()

    img_list = glob(metadata_dir + '*')
    img_list.sort()

    for img in img_list:
        img_json = ImageJSON()
        img_json.load(img)
        obj = img_json.get_objects()
        for o in obj:
            wid = v.get_id_or_add(o.synsets)
            v.wcount[wid] = v.wcount.get(wid, 0) + 1

    for wid, num in v.wcount.items():
        if num < min_tf: continue
        new_wid = vocab.get_id_or_add(v.get_word(wid))
        vocab.wcount[new_wid] = num
    del v

    vocab.save_vocab(save_dir=os.path.join(make_dir(vocab_dir), 'vocab.txt'))
    vocab.save_word_count(save_dir=os.path.join(vocab_dir, 'wcount.txt'))


def generate_co_occurrence_matrix(save_name,
                                  context_window=4,
                                  metadata_dir='../data/vg-metadata/by-id-train/',
                                  vocab_dir='../result/vocab/',
                                  matrix_dir='../result/matrix/',
                                  weighted=True):
    print('Generate co-occurrence matrix with context window size [{}]\n'.format(context_window))

    # if check_file_in_dir(check_dir=matrix_dir, check_file='matrix.npz'): return -1

    vocab = Vocabulary()
    vocab.load_vocab(os.path.join(vocab_dir, 'vocab.txt'))

    co_occur_matrix = CoOccurrenceMatrix(dim=len(vocab.word))

    img_list = glob(os.path.join(metadata_dir, '*'))
    img_list.sort()

    if weighted:
        co_occur_value = [1.0 / (i + 1) for i in range(context_window)]
    else:
        co_occur_value = [1 for i in range(context_window)]

    for img in img_list:
        img_json = ImageJSON()
        img_json.load(img)
        obj = img_json.get_objects()

        for t_idx in range(len(obj)):
            t_synset = obj[t_idx].synsets
            if vocab.get_id(t_synset) < 0: continue

            c_synset_list = list()
            c_dis_list = list()

            for before_t in range(0, t_idx):
                if vocab.get_id(obj[before_t].synsets) < 0: continue
                if obj[before_t].synsets == t_synset: continue
                c_synset_list.append(obj[before_t].synsets)
                c_dis_list.append(perceptual_dis(obj[t_idx], obj[before_t]))
            for after_t in range(t_idx + 1, len(obj)):
                if vocab.get_id(obj[after_t].synsets) < 0: continue
                if obj[after_t].synsets == t_synset: continue
                c_synset_list.append(obj[after_t].synsets)
                c_dis_list.append(perceptual_dis(obj[t_idx], obj[after_t]))

            c_distance_argsorted = np.array(c_dis_list).argsort()
            n_c = len(c_distance_argsorted)

            if n_c < context_window:
                n_iter = n_c
            else:
                n_iter = context_window

            for c_idx in range(n_iter):
                co_occur_matrix.row.append(vocab.get_id(t_synset))
                co_occur_matrix.col.append(vocab.get_id(c_synset_list[c_distance_argsorted[c_idx]]))
                co_occur_matrix.data.append(co_occur_value[c_idx])

                co_occur_matrix.row.append(vocab.get_id(c_synset_list[c_distance_argsorted[c_idx]]))
                co_occur_matrix.col.append(vocab.get_id(t_synset))
                co_occur_matrix.data.append(co_occur_value[c_idx])

    co_occur_matrix.init_matrix()
    co_occur_matrix.save_matrix(save_dir=os.path.join(make_dir(matrix_dir), 'matrix.npz'.format()))


def train_vce(no_components=32,
              epochs=200,
              vocab_dir='../result/vocab/',
              matrix_dir='../result/matrix/',
              embedding_model_dir='../result/embedding-model/',
              vce_dir='../result/vce/'):
    print('Train [{}] dimension Visual-Context-Embedding vectors with [{}] epochs.'.format(no_components, epochs))    

    vocab = Vocabulary()
    vocab.load_vocab(os.path.join(vocab_dir, 'vocab.txt'))

    co_occur_matrix = CoOccurrenceMatrix(dim=len(vocab.word))
    co_occur_matrix.load_matrix(load_dir=os.path.join(matrix_dir, 'matrix.npz'))
    # co_occur_matrix.load_matrix(load_dir='../result/matrix-min010-bbox.npz')

    embedding_model = Glove(no_components=no_components, learning_rate=0.05, max_count=500)

    embedding_model.fit(co_occur_matrix.matrix.tocoo().astype(np.float64), epochs=epochs, no_threads=4, verbose=True)
    embedding_model.add_dictionary(vocab.id)

    model_save_name = os.path.join(embedding_model_dir, 'vce{}-model.pickle'.format(no_components))
    embedding_model.save(model_save_name)

    vce = l2_normalize_embedding(embedding_model.word_vectors, dim=no_components)
    vce_save_name = os.path.join(vce_dir, 'vce{}.npy'.format(no_components, save_name))
    np.save(vce_save_name, vce)

    print('{} saved.\n'.format(vce_save_name))


def train_vce_ablation(no_components=32,
                       epochs=100,
                       vocab_dir='../result/vocab/',
                       matrix_dir='../result/ablation/matrix/',
                       embedding_model_dir='../result/ablation/embedding-model/',
                       vce_dir='../result/ablation/vce/'):
    print('Train [{}] dimension Visual-Context-Embedding vectors with [{}] epochs.'.format(no_components, epochs))    

    vocab = Vocabulary()
    vocab.load_vocab(os.path.join(vocab_dir, 'vocab.txt'))

    co_occur_matrix = CoOccurrenceMatrix(dim=len(vocab.word))
    matrix_list = glob(matrix_dir+'*')
    for matrix in matrix_list:
        save_name = os.path.splitext(matrix)[0].split('_')[-1]

        co_occur_matrix.load_matrix(load_dir=matrix)

        embedding_model = Glove(no_components=no_components, learning_rate=0.05, max_count=500)

        embedding_model.fit(co_occur_matrix.matrix.tocoo().astype(np.float64), epochs=epochs, no_threads=4, verbose=True)
        embedding_model.add_dictionary(vocab.id)

        model_save_name = os.path.join(embedding_model_dir, 'vce{}-{}.pickle'.format(no_components, save_name))
        embedding_model.save(model_save_name)

        vce = l2_normalize_embedding(embedding_model.word_vectors, dim=no_components)
        vce_save_name = os.path.join(vce_dir, 'vce{}-{}.npy'.format(no_components, save_name))
        np.save(vce_save_name, vce)

        print('{} saved.\n'.format(vce_save_name))


def split_seen_unseen(dataset_name, 
                      dim,
                      vocab_dir='../result/vocab/',
                      vce_dir='../result/vce/',
                      split_dir='../result/vce-split/'):
    vce_name = os.path.join(vce_dir, 'vce{}.npy'.format(dim))
    vce = np.load(vce_name)

    vocab = Vocabulary()
    vocab_name = os.path.join(vocab_dir, 'vocab.txt')
    vocab.load_vocab(vocab_name)

    with open('../data/{}/seen-word-synset.txt'.format(dataset_name), 'r') as fr:
        seen_list = [line.replace('\n', '') for line in fr.readlines()]

    with open('../data/{}/unseen-word-synset.txt'.format(dataset_name), 'r') as fr:
        unseen_list = [line.replace('\n', '') for line in fr.readlines()]
    
    seen_list = sync_namespace(seen_list, dataset_name)
    unseen_list = sync_namespace(unseen_list, dataset_name)

    n_seen = len(seen_list)
    n_unseen = len(unseen_list)

    vce_seen = np.empty((dim, n_seen))
    vce_unseen = np.empty((dim, n_unseen))

    for i in range(n_seen):
        vce_seen[:, i] = vce[:, vocab.get_id(seen_list[i])]

    for j in range(n_unseen):
        vce_unseen[:, j] = vce[:, vocab.get_id(unseen_list[j])]

    save_name_seen = os.path.join(split_dir, 'vce{}-{}-seen.npy'.format(dim, dataset_name))
    save_name_unseen = os.path.join(split_dir, 'vce{}-{}-unseen.npy'.format(dim, dataset_name))
    
    np.save(save_name_seen, vce_seen)
    np.save(save_name_unseen, vce_unseen)


def split_seen_unseen_ablation(dataset_name,
                               dim,
                               vocab_dir='../result/vocab/',
                               vce_dir='../result/ablation/vce/',
                               split_dir='../result/ablation/vce-split/'):
    
    vocab = Vocabulary()
    vocab_name = os.path.join(vocab_dir, 'vocab.txt')
    vocab.load_vocab(vocab_name)
    
    ablation_list = glob(vce_dir+'*')
    for ab in ablation_list:
        save_name = os.path.splitext(ab)[0].split('-')[-1]
        vce = np.load(ab)

        with open('../data/{}/seen-word-synset.txt'.format(dataset_name), 'r') as fr:
            seen_list = [line.replace('\n', '') for line in fr.readlines()]

        with open('../data/{}/unseen-word-synset.txt'.format(dataset_name), 'r') as fr:
            unseen_list = [line.replace('\n', '') for line in fr.readlines()]


        seen_list = sync_namespace(seen_list, dataset_name)
        unseen_list = sync_namespace(unseen_list, dataset_name)

        n_seen = len(seen_list)
        n_unseen = len(unseen_list)

        vce_seen = np.empty((dim, n_seen))
        vce_unseen = np.empty((dim, n_unseen))

        for i in range(n_seen):
            vce_seen[:, i] = vce[:, vocab.get_id(seen_list[i])]


        for j in range(n_unseen):
            vce_unseen[:, j] = vce[:, vocab.get_id(unseen_list[j])]

        save_name_seen = os.path.join(split_dir, 'vce{}-{}-{}-seen.npy'.format(dim, save_name, dataset_name))
        save_name_unseen = os.path.join(split_dir, 'vce{}-{}-{}-unseen.npy'.format(dim, save_name, dataset_name))
        
        np.save(save_name_seen, vce_seen)
        np.save(save_name_unseen, vce_unseen)


def sync_namespace(synset_list, dataset_name):
    if dataset_name == 'vg':
        # key, value = seen(unseen)-word-synset.txt, vocab.txt
        sync_dict = {
            'monitor.n.04': 'monitor.n.05',
            'cow.n.01': 'cow.n.02', 
            'toaster.n.01': 'toaster.n.02', 
            'book.n.01': 'book.n.02',
            'glove.n.01': 'glove.n.02',
            'hotdog.n.01': 'hotdog.n.02',
            'plant.n.01': 'plant.n.02'}
    elif dataset_name == 'coco':
        sync_dict = {
            'orange.n.01': 'fruit.n.01',
            'television_monitor.n.01': 'monitor.n.05'
        }
    else:
        raise ValueError('dataset_name should be either "vg" or "coco".')

    n_synset = len(synset_list)

    for i in range(n_synset):
        if synset_list[i] in sync_dict.keys():
            synset_list[i] = sync_dict[synset_list[i]]
    
    return synset_list


if __name__=='__main__':
    print('# main')
    print('# ---------------------------------------------------------------------------------------------------------\n')
    
    # # Construct Visual-Context-Embedding
    # json_preprocessing()
    # initialize_vocab()
    # generate_co_occurrence_matrix(save_name='matrix.npz')

    # for dim in [16, 32, 64, 128]:
    #     train_vce(no_components=dim)

    # for dim in [16, 32, 64, 128]:
    #     for d_name in ['vg', 'coco']:
    #         split_seen_unseen(dataset_name=d_name, dim=dim)

    train_vce_ablation()
    for d_name in ['vg', 'coco']:
        split_seen_unseen_ablation(dataset_name=d_name, dim=32)


    # # Visualize Embedding Space
    # src_dir='../result/vce/'
    # for n_dim in [16, 32, 64, 128]:
    #     for dataset in ['vg', 'coco']:
    #         vce_seen = np.load(os.path.join(src_dir, 'vce{}-{}-seen.npy'.format(n_dim, dataset)))
    #         vce_unseen = np.load(os.path.join(src_dir, 'vce{}-{}-unseen.npy'.format(n_dim, dataset)))
    #         visualize_vector(vce_seen)
    #         visualize_vector(vce_unseen)
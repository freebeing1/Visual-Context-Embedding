import json

from collections import OrderedDict
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz


class CoOccurrenceMatrix:
    def __init__(self, dim):
        self.matrix = None
        self.row = list()
        self.col = list()
        self.data = list()
        self.dim = dim

    def init_matrix(self):
        self.matrix = csr_matrix((self.data, (self.row, self.col)), shape=(self.dim, self.dim))

    def save_matrix(self, save_dir='matrix.npz'):
        save_npz(save_dir, self.matrix)

    def load_matrix(self, load_dir='matrix.npz'):
        self.matrix = load_npz(load_dir)


class Vocabulary:
    def __init__(self):
        self.id = OrderedDict()
        self.word = list()
        self.wcount = OrderedDict()

    def get_id_or_add(self, word):
        if word in self.id: return self.id[word]
        self.id[word] = len(self.id)
        self.word.append(word)
        return len(self.id) - 1

    def get_id(self, word):
        if word in self.id: return self.id[word]
        return -1

    def get_word(self, id):
        return self.word[id]

    def load_vocab(self, load_dir='vocab.txt'):
        with open(load_dir, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line == '\n': continue
                self.get_id_or_add(line.replace('\n',''))

    def load_word_count(self, load_dir='word-count.txt'):
        with open(load_dir, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if line == '\n': continue
                self.wcount[self.id[line.replace('\n', '').split()[0]]] = int(line.replace('\n', '').split()[1])

    def save_vocab(self, save_dir='vocab.txt'):
        with open(save_dir, 'w') as fw:
            for w in self.word:
                fw.write('{}\n'.format(w))

    def save_word_count(self, save_dir='word-count.txt'):
        # print(len(self.word))
        # print(len(self.wcount))
        with open(save_dir, 'w') as fw:
            for i in range(len(self.word)):
                fw.write('{} {}\n'.format(self.word[i], self.wcount[i]))


class Synset:
    def __init__(self, synsets, x, y, w, h):
        self.synsets = synsets
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = (self.x + self.w/2, self.y + self.h/2)


class ImageJSON:
    def __init__(self, image_id=None):
        self.image_id = image_id
        self.objects = list()

    def init_objects(self, objects):
        for obj in objects:
            if len(obj['synsets']) == 0: continue
            else: self.objects.append(Synset(synsets=obj['synsets'][0], x=obj['x'], y=obj['y'], w=obj['w'], h=obj['h']))

    def get_id(self):
        return self.image_id

    def get_objects(self):
        return self.objects

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def load(self, fname):
        jsondata = json.load(open(fname, 'r', encoding='utf-8'))
        self.image_id = jsondata['image_id']
        self.init_objects(jsondata['objects'])

    def save(self, path='by-id/'):
        with open(path + '{}'.format(str(self.image_id).zfill(8) + '.json'), 'w') as fw:
            fw.write(self.to_json())

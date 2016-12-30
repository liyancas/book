import collections
import cPickle
import py_paddle.swig_paddle as api
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy
from paddle.trainer.PyDataProvider2 import *
from paddle.trainer_config_helpers import *
from py_paddle.trainer import *

TRAIN_FILE = "./data/simple-examples/data/ptb.train.txt"
TEST_FILE = './data/simple-examples/data/ptb.test.txt'


class WordDictParser(object):
    """
    :type __word_dict__: dict
    """

    def __init__(self):
        self.__word_dict__ = collections.defaultdict(int)
        self.__word_dict__["<s>"] = -1
        self.__word_dict__["<e>"] = -2

    def parse_file(self, fn):
        with open(fn, 'r') as f:
            for line in f:
                for word in line.split():
                    word = word.strip()
                    self.__word_dict__[word] += 1

    def done(self):
        words = []
        for k in self.__word_dict__:
            words.append((k, self.__word_dict__[k]))

        words.sort(cmp=lambda a, b: a[1] - b[1])

        retv = dict()
        for i, (word, word_count) in enumerate(words):
            retv[word] = {
                'index': i,
                'word_count': word_count
            }
        return retv


try:
    with open('word.dict', 'rb') as f:
        WORD_DICT = cPickle.load(f)
except:
    parser = WordDictParser()
    parser.parse_file(TRAIN_FILE)
    parser.parse_file(TEST_FILE)
    WORD_DICT = parser.done()
    with open("word.dict", 'wb') as f:
        cPickle.dump(WORD_DICT, f)

WORD_DIM = len(WORD_DICT)
EMB_DIM = 32


@network(inputs={
    'w_4': integer_value(WORD_DIM),
    'w_3': integer_value(WORD_DIM),
    'w_2': integer_value(WORD_DIM),
    'w_1': integer_value(WORD_DIM),
    'w': integer_value(WORD_DIM)
}, learning_rate=1e-4, batch_size=128,
    learning_method=RMSPropOptimizer())
def ngram_network(w, w_1, w_2, w_3, w_4):
    words = [w_1, w_2, w_3, w_4]
    embeddings = [embedding_layer(input=each_word,
                                  size=EMB_DIM,
                                  param_attr=ParameterAttribute(
                                      name='embedding', sparse_update=True)) for
                  each_word in words]
    huge_embedding = concat_layer(input=embeddings)
    hidden = fc_layer(input=huge_embedding, size=256)
    predict = fc_layer(input=hidden, size=WORD_DIM, act=SoftmaxActivation())
    return classification_cost(input=predict, label=w)


ngram = ngram_network()


@ngram.provider()
def process(settings, filename, **kwargs):
    with open(filename, 'r') as f:
        for line in f:
            words = [w.strip() for w in line.split()]
            words = ['<s>'] * 4 + words + ['<e>']
            words = [WORD_DICT[w]["index"] for w in words]

            for i in xrange(4, len(words)):
                ret_dict = {}
                for j in xrange(1, 5):
                    ret_dict["w_%d" % j] = words[i - j]
                ret_dict['w'] = words[i]
                yield ret_dict


runner = RunnerBuilder(network=ngram, device_count=2).with_std_local_trainer(
    method=process, file_list=[TRAIN_FILE]).with_std_tester(
    method=process, file_list=[TEST_FILE]).build()


def save_plotting(pass_id, vocabulary, wv):
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(wv)
    with open('embedding_%d.plt' % pass_id, 'rb') as f:
        cPickle.dump({
            'embs': Y,
            'words': vocabulary
        }, f)


with runner:
    words = []

    for k in WORD_DICT:
        words.append((k, WORD_DICT[k]['word_count']))

    words.sort(cmp=lambda a, b: a[1] > b[1])

    words = words[:100]

    words = [w[0] for w in words]

    for pass_id in xrange(100):
        context = runner.run_one_pass()
        context = ContextWrapper(context)
        params = context.gradient_machine().getParameters()
        for param in params:
            assert isinstance(param, api.Parameter)
            if param.getName() == 'embedding':
                emb = param.getBuf(api.PARAMETER_VALUE)
                assert isinstance(emb, api.Vector)
                embNp = emb.copyToNumpyArray()
                embNp = embNp.reshape((len(embNp) / EMB_DIM, EMB_DIM))

                we = numpy.zeros((len(words), EMB_DIM))

                for i, w in enumerate(words):
                    we[i] = embNp[WORD_DICT[w]['index']]

                save_plotting(pass_id, words, we)

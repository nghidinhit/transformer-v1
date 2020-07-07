from hyperparams import Hyperparams as params

import numpy as np
import codecs
import random


def load_source_vocab():
    vocab = [line.split()[0] for line in codecs.open(params.src_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= params.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_target_vocab():
    vocab = [line.split()[0] for line in codecs.open(params.tgt_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= params.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_vocab(vocab_file):
    vocab = [line.split()[0] for line in codecs.open(vocab_file, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= params.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def convert_word2idx(word2idx, samples):
    samples_idxes = list()
    for source_sent in samples:
        sample_idx = list()
        for word in (source_sent + " </s>").split():
            if word in word2idx:
                sample_idx.append(word2idx[word])
            else:
                sample_idx.append(word2idx['<unk>'])

        if len(sample_idx) <= params.maxlen:
            samples_idxes.append(np.array(sample_idx))

    # Pad
    sample_idx_pad = np.zeros([len(samples_idxes), params.maxlen], np.int32)
    for i, sample_idx in enumerate(samples_idxes):
        sample_idx_pad[i] = np.lib.pad(sample_idx, [0, params.maxlen - len(sample_idx)], 'constant', constant_values=(0, 0))

    return sample_idx_pad


def create_data(source_sents, target_sents): 
    source2idx, idx2source = load_vocab(params.src_vocab)
    target2idx, idx2target = load_vocab(params.tgt_vocab)
    
    # Index
    source_idx, target_idx, source_text, target_text = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # x = [source2idx[word] for word in (source_sent + " </S>").split()] # 1: OOV, </S>: End of Text
        # y = [target2idx[word] for word in (target_sent + " </S>").split() if word in target2idx else target2idx['<UNK>']]

        x = list()
        for word in (source_sent + " </s>").split():
            if word in source2idx:
                x.append(source2idx[word])
            else:
                x.append(source2idx['<unk>'])

        y = list()
        for word in (target_sent + " </s>").split():
            if word in target2idx:
                y.append(target2idx[word])
            else:
                y.append(target2idx['<unk>'])

        if max(len(x), len(y)) <= params.maxlen:
            source_idx.append(np.array(x))
            target_idx.append(np.array(y))
            source_text.append(source_sent)
            target_text.append(target_sent)

    # Pad      
    source_idxes = np.zeros([len(source_idx), params.maxlen], np.int32)
    target_idxes = np.zeros([len(target_idx), params.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(source_idx, target_idx)):
        source_idxes[i] = np.lib.pad(x, [0, params.maxlen - len(x)], 'constant', constant_values=(0, 0))
        target_idxes[i] = np.lib.pad(y, [0, params.maxlen - len(y)], 'constant', constant_values=(0, 0))
    
    return source_idxes, target_idxes, source_text, target_text


def create_data_word_label(source_sents, target_sent_words, target_sent_labels):
    source2idx, idx2source = load_vocab(params.src_vocab_word_label)
    target2idx_word_label, idx2target_word_label = load_vocab(params.tgt_vocab_word_label)
    target2idx_word, idx2target_word = load_vocab(params.tgt_vocab_word)
    target2idx_label, idx2target_label = load_vocab(params.tgt_vocab_label)

    # Index
    source_idx, target_idx_word, target_idx_label, source_text, target_text_words, target_text_labels = [], [], [], [], [], []

    for source_sent, target_sent_word, target_sent_label in zip(source_sents, target_sent_words, target_sent_labels):

        x = list()
        for word in (source_sent + " </s>").split():
            if word in source2idx:
                x.append(source2idx[word])
            else:
                x.append(source2idx['<unk>'])

        y_word = list()
        for word in (target_sent_word + " </s>").split():
            if word in target2idx_word:
                y_word.append(target2idx_word[word])
            else:
                y_word.append(target2idx_word['<unk>'])

        y_label = list()
        for label in (target_sent_label + " </s>").split():
            if label in target2idx_label:
                y_label.append(target2idx_label[label])
            else:
                y_label.append(target2idx_label['<unk>'])

        if max(len(x), len(y_word), len(y_label)) <= params.maxlen:
            source_idx.append(np.array(x))
            target_idx_word.append(np.array(y_word))
            target_idx_label.append(np.array(y_label))
            source_text.append(source_sent)
            target_text_words.append(target_sent_word)
            target_text_labels.append(target_sent_label)

    # Pad
    source_idxes = np.zeros([len(source_idx), params.maxlen], np.int32)
    target_idx_words = np.zeros([len(target_idx_word), params.maxlen], np.int32)
    target_idx_labels = np.zeros([len(target_idx_label), params.maxlen], np.int32)
    for i, (x, y_word, y_label) in enumerate(zip(source_idx, target_idx_word, target_idx_label)):
        source_idxes[i] = np.lib.pad(x, [0, params.maxlen - len(x)], 'constant', constant_values=(0, 0))
        target_idx_words[i] = np.lib.pad(y_word, [0, params.maxlen - len(y_word)], 'constant', constant_values=(0, 0))
        target_idx_labels[i] = np.lib.pad(y_label, [0, params.maxlen - len(y_label)], 'constant', constant_values=(0, 0))

    return source_idxes, target_idx_words, target_idx_labels, source_text, target_text_words, target_text_labels


def load_train_data():
    if params.is_lower:
        src_sents = [line.lower() for line in open(params.source_train, 'r').read().split("\n")]
        tgt_sents = [line.lower() for line in open(params.target_train, 'r').read().split("\n")]
    else:
        src_sents = [line for line in open(params.source_train, 'r').read().split("\n")]
        tgt_sents = [line for line in open(params.target_train, 'r').read().split("\n")]

    source_idxes, target_idxes, source_text, target_text = create_data(src_sents, tgt_sents)
    return source_idxes, target_idxes


def load_test_data():
    if params.is_lower:
        src_sents = [line.lower() for line in open(params.source_test, 'r').read().split("\n")]
        tgt_sents = [line.lower() for line in open(params.target_test, 'r').read().split("\n")]
    else:
        src_sents = [line for line in open(params.source_test, 'r').read().split("\n")]
        tgt_sents = [line for line in open(params.target_test, 'r').read().split("\n")]

    source_idxes, target_idxes, source_text, target_text = create_data(src_sents, tgt_sents)
    return source_idxes, source_text, target_text  # (1064, 150)


def load_data(src, tgt, num_sample):
    # if params.is_lower:
    #     src_sents = [line.lower() for line in open(src, 'r').read().split("\n")]
    #     tgt_sents = [line.lower() for line in open(tgt, 'r').read().split("\n")]
    # else:
    #     src_sents = [line for line in open(src, 'r').read().split("\n")]
    #     tgt_sents = [line for line in open(tgt, 'r').read().split("\n")]

    src_sents, tgt_sents = list(), list()

    for line in open(src, 'r').read().split("\n"):
        if len(src_sents) < num_sample:
            if params.is_lower:
                src_sents.append(line.lower())
            else:
                src_sents.append(line)
        else: break

    for line in open(tgt, 'r').read().split("\n"):
        if len(tgt_sents) < num_sample:
            if params.is_lower:
                tgt_sents.append(line.lower())
            else:
                tgt_sents.append(line)
        else: break

    source_idxes, target_idxes, source_text, target_text = create_data(src_sents, tgt_sents)
    return source_idxes, target_idxes, source_text, target_text


def load_data_word_label(src, tgt_word, tgt_label, num_sample):

    src_sents, tgt_sent_words, tgt_sent_labels = list(), list(), list()

    for line in open(src, 'r').read().split("\n"):
        if len(src_sents) < num_sample:
            if params.is_lower:
                src_sents.append(line.lower())
            else:
                src_sents.append(line)
        else: break

    for line in open(tgt_word, 'r').read().split("\n"):
        if len(tgt_sent_words) < num_sample:
            if params.is_lower:
                tgt_sent_words.append(line.lower())
            else:
                tgt_sent_words.append(line)
        else: break

    for line in open(tgt_label, 'r').read().split("\n"):
        if len(tgt_sent_labels) < num_sample:
            if params.is_lower:
                tgt_sent_labels.append(line.lower())
            else:
                tgt_sent_labels.append(line)
        else: break

    source_idxes, target_idx_words, target_idx_labels, source_text, target_text_words, target_text_labels = create_data_word_label(src_sents, tgt_sent_words, tgt_sent_labels)
    return source_idxes, target_idx_words, target_idx_labels, source_text, target_text_words, target_text_labels


def get_batch_indices(total_length, batch_size):
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index: current_index + batch_size], current_index



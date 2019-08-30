import os
from hyperparams import Hyperparams as param
from collections import Counter
from string import punctuation
from sklearn.utils import shuffle
import pandas as pd
import csv
import re
import torch


def make_vocab(fpath, fname):
    text = open(fpath, 'r').read()
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('vocab'):
        os.mkdir('vocab')
    with open('vocab/{}'.format(fname), 'w') as fout:
        fout.write("{}\t1000000\n{}\t1000000\n{}\t1000000\n{}\t1000000\n".format("<pad>", "<unk>", "<s>", "</s>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write("{}\t{}\n".format(word, cnt))
        # fout.write("{}\t{}\n".format('_', '100'))


def process_g2p_english(f_file):
    graphes, phonemes = list(), list()
    fo = open(f_file, 'r')
    count = 0
    for line in fo:
        count += 1
        if count % 1000:
            print('count = %d' % count)
        graph = line.split()[0]
        phoneme = ' '.join(line.split()[1:])

        if graph[0] in punctuation:
            graph = graph[1:]

        graph = re.sub('\(1\)', '', graph)
        graph = re.sub('\(2\)', '', graph)
        graph = re.sub('\(3\)', '', graph)
        if graph not in graphes:
            graphes.append(graph)
            phonemes.append(phoneme)

    return graphes, phonemes


def save_list(l, file):
    fo = open(file, 'w')
    for e in l:
        fo.write(e + '\n')


def preprocess(text):
    vnese_lower = 'aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵdđ'
    vnese_upper = 'AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴDĐ'
    text = re.sub('\([a-zA-Z_' +vnese_lower +vnese_upper + ']*\)', '', text)
    return text


def load_g2p(f_g2p):
    fo = open(f_g2p, 'r')
    lines = list()
    for line in fo:
        input_text = ' '.join(list(line.split()[0]))
        target_text = ' '.join(line.split()[1:])
        lines.append(input_text + '\t' + target_text)

    return lines


def load_g2p_english(graph_file, phoneme_file):
    lines = list()
    f_graph = open(graph_file, 'r')
    f_phoneme = open(phoneme_file, 'r')
    graphes = f_graph.readlines()
    phonemes = f_phoneme.readlines()
    for i in range(len(graphes)):
        line = graphes[i] + '\t' + phonemes[i]
        lines.append(line)

    return lines


def load_foreign_words(f_foreign):
    csv_writer = csv.writer(open('sample_synthesize.csv', 'w'), delimiter='\t')
    csv_writer.writerow(['written', 'spoken'])
    df = pd.read_csv(f_foreign)
    words_foreign = df.word.values.tolist()
    trans_foreign = df.transcription.values.tolist()
    lines = list()
    for i in range(len(words_foreign)):
        input_text = str(words_foreign[i]).lower()
        target_text = str(trans_foreign[i]).lower()
        target_text = target_text.replace('-','_')
        target_text = target_text.replace(' ', '$')
        target_text = target_text.replace('_', ' ')
        target_text = target_text.replace('$', ' _ ')
        input_text = ' '.join(preprocess(input_text))
        target_text = preprocess(target_text)
        # input_words = input_text.split()
        # target_words = target_text.split()
        # if len(input_words) == len(target_words):
        line = input_text + '\t' + target_text
        lines.append(line)

    return lines


def load_foreign_words_v2(f_foreign):
    df = pd.read_csv(f_foreign)
    words_foreign = df.word.values.tolist()
    trans_foreign = df.transcription.values.tolist()
    lines = list()
    line_errors = list()
    for i in range(len(words_foreign)):
        input_text = str(words_foreign[i]).lower()
        target_text = str(trans_foreign[i]).lower()
        target_text = target_text.replace('-','_')
        # target_text = target_text.replace(' ', '$')
        # target_text = target_text.replace('_', ' ')
        # target_text = target_text.replace('$', ' _ ')
        input_text = preprocess(input_text)
        target_text = preprocess(target_text)
        input_words = input_text.split()
        target_words = target_text.split()
        if len(input_words) == len(target_words):
            for j in range(len(input_words)):
                target_words[j] = target_words[j].replace('_', ' ')
                line = ' '.join(list(input_words[j])) + '\t' + target_words[j]
                if line not in lines:
                    lines.append(line)
        else:
            target_text = target_text.replace('-','_')
            line_errors.append(input_text + '\t' + target_text)
            lines.append(' '.join(list(input_text)) + '\t' + target_text)

    return lines, line_errors


def load_tone_data(csv_file):
    df = pd.read_csv(csv_file, delimiter='\t')
    line_no_tone = df['no_tone'].values.tolist()
    line_with_tone = df['tone'].values.tolist()
    lines = list()
    for i in range(len(line_no_tone)):
        lines.append(str(line_no_tone[i]) + '\t' + str(line_with_tone[i]))
    return lines


def load_seq2seq(csv_file, tag_col, src_col, tgt_col):
    df = pd.read_csv(csv_file, delimiter='\t')
    tag = df[tag_col].values.tolist()
    src = df[src_col].values.tolist()
    tgt = df[tgt_col].values.tolist()
    lines = list()
    for i in range(len(src)):
        # lines.append(str(tag[i].strip()) + ' ' + str(src[i]) + '\t' + str(tgt[i]))
        lines.append(str(tag[i].strip()) + ' ' + str(tgt[i]) + '\t' + str(src[i]))
    return lines


def split_train_test(lines, path, train_ratio=0.8, val_ratio=0.1):
    lines = shuffle(lines)
    train_size = int(len(lines) * train_ratio)
    val_size = int(len(lines) * val_ratio)
    print('train size: ', train_size)
    print('val size: ', val_size)
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]
    save_data(train_data, 'train', path)
    save_data(val_data, 'val', path)
    save_data(test_data, 'test', path)


def save_data(lines, type, path):
    src = list()
    tgt = list()
    f_src = open(path + type + '.src.txt', 'w')
    f_tgt = open(path + type + '.tgt.txt', 'w')
    for line in lines:
        pair = line.split('\t')
        f_src.write(pair[0].strip() + '\n')
        f_tgt.write(pair[1].strip() + '\n')
        src.append(pair[0])
        tgt.append(pair[1])


def read_data(data_path):
    data = torch.load(data_path)
    dict = data['dict']
    tgt_word2idx = dict['tgt']
    print(tgt_word2idx)


def save_file(lines, path, f_name):
    f_out = open(path + f_name, 'w')
    for line in lines:
        f_out.write(line + '\n')


def word2char(f_word, f_char):
    fo = open(f_word, 'r')
    fw = open(f_char, 'w')
    for line in fo:
        fw.write(' '.join(list(line)))


def save_csv(l1, l2, csv_file):
    csv_writer = csv.writer(open(csv_file, 'w'), delimiter='\t')
    csv_writer.writerow(['graph_word', 'graph_char', 'phoneme'])
    l1_char = list()
    for e in l1:
        l1_char.append(' '.join(list(e)))
    for i in range(len(l1)):
        csv_writer.writerow([l1[i], l1_char[i], l2[i]])


if __name__ == '__main__':
    # make_vocab(param.source_train, param.src_vocab)
    # make_vocab(param.target_train, param.tgt_vocab)
    # graphes, phonemes = process_g2p_english(param.g2p_english)
    # save_list(graphes, param.graph_english)
    # save_list(phonemes, param.phoneme_english)
    # save_csv(graphes, phonemes, 'corpora/g2p_english/g2p_english.csv')
    # word2char(param.graph_english, param.graph_english_char)

    lines = load_tone_data('corpora/addtone/train_tokenizer.csv')
    # lines = load_foreign_words('dataset/norm_foreign/foreign.csv')
    # lines, line_errors = load_foreign_words_v2('dataset/norm_foreign/foreign.csv')
    # save_file(line_errors, 'dataset/norm_foreign_v2/', 'line_errors.txt')
    # lines = load_seq2seq('dataset/auto_upper_punct_seq2seq/shard_0000_auto_upper_punct_seq2seq.csv', 'src', 'tgt')
    # lines = load_seq2seq('dataset/spoken2written/norm_seq2seq.csv', 'tag', 'written', 'spoken')
    # lines = load_g2p('dataset/g2p/foreign-lexicon-13k_one_type.lex')
    # lines = load_g2p_english(param.graph_english_char, param.phoneme_english)
    split_train_test(lines, 'corpora/addtone/')
    # read_data('dataset/norm_foreign/norm_foreign.pkl')

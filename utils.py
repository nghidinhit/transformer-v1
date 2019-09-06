import os
from hyperparams import Hyperparams as param
from collections import Counter
from string import punctuation
from sklearn.utils import shuffle
from nltk import word_tokenize
import pandas as pd
import csv
import re
import torch
import codecs
import string


punct_dict = {';':'.', ':':',', '!' : '.'}
special_char_dict_v2 = {',':'COMMA', '.':'PERIOD', '?':'QUESTION'}


def make_vocab(fpath, fname):
    text = open(fpath, 'r').read()
    words = text.split()
    word2cnt = Counter(words)
    # if not os.path.exists('vocab'):
    #     os.mkdir('vocab')
    with open('{}'.format(fname), 'w') as fout:
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
        graph = line.split('')[0]
        phoneme = ' '.join(line.split('')[1:])

        if graph[0] in punctuation:
            graph = graph[1:]

        graph = re.sub('\(1\)', '', graph)
        graph = re.sub('\(2\)', '', graph)
        graph = re.sub('\(3\)', '', graph)
        if graph not in graphes:
            graphes.append(graph)
            phonemes.append(phoneme)

    return graphes, phonemes


def process_g2p_vnmese(f_file):
    graphes, phonemes = list(), list()
    fo = open(f_file, 'r')
    count = 0
    for line in fo:
        count += 1
        if count % 1000:
            print('count = %d' % count)
        graph = line.split(' ', 1)[0].strip()
        phoneme = line.split(' ', 1)[1].strip()
        phoneme = phoneme.replace(' ', ' $ ').replace('|', ' | ')

        if graph in punctuation:
            continue

        graph = re.sub('\(1\)', '', graph)
        graph = re.sub('\(2\)', '', graph)
        graph = re.sub('\(3\)', '', graph)

        # graph_ch = list()
        # for c in list(graph):
        #     if c in punctuation:
        #         graph_ch.append(' ')
        #     else: graph_ch.append(c)
        # graph = ''.join(graph_ch)

        if graph not in graphes:
            graphes.append(graph)
            phonemes.append(phoneme)
    return graphes, phonemes


def save_list(l, file):
    fo = open(file, 'w')
    for e in l:
        fo.write(e.strip() + '\n')


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


def load_csv_2_cols(csv_file, col1, col2):
    df = pd.read_csv(csv_file, delimiter='\t')
    col1_data = df[col1].values.tolist()
    col2_data = df[col2].values.tolist()
    lines = list()
    for i in range(len(col1_data)):
        lines.append(str(col1_data[i]) + '\t' + str(col2_data[i]))
    return lines


def load_csv_3_cols(csv_file, col1, col2, col3):
    df = pd.read_csv(csv_file, names=[col1, col2, col3], delimiter='\t')
    line_col1 = df[col1].values.tolist()
    line_col2 = df[col2].values.tolist()
    line_col3 = df[col3].values.tolist()
    lines = list()
    for i in range(len(line_col1)):
        lines.append(str(line_col1[i]) + ' ' + str(line_col2[i]) + '\t' + str(line_col3[i]))
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
        l1_char.append(' '.join(list(e.strip())))
    for i in range(len(l1)):
        csv_writer.writerow([l1[i], l1_char[i], l2[i].strip()])


def revert_norm_word(str):
    words = str.split()
    for i in range(1, len(words)):
        if words[i] in special_char_dict_v2.keys():
            words[i - 1] = words[i - 1] + words[i]
            words[i] = ''

    return ' '.join(words)


# remove punctuation exclude .,?
def remove_puntuation(str):
    tokens = list()
    for token in str.split():
        if token in punct_dict.keys():
            tokens.append(punct_dict[token])
        elif token in special_char_dict_v2.keys():
            tokens.append(token)
        elif token in punctuation:
            tokens.append('')
        else:
            tokens.append(token)

    input_str = ' '.join(tokens)
    input_str = re.sub('\.+\s*[\.*|,*|\?*]', '.', input_str)
    input_str = re.sub(',+\s*[\.*|,*|\?*]', ',', input_str)
    input_str = re.sub('\?+\s*[\.*|,*|\?*]', ',', input_str)
    return input_str


# manually tokenize
def norm_word(word):
    char = list(word)
    pre_is_word = False
    pre_is_punct = False
    for i in range(len(char)):
        if char[i] in punctuation:
            if pre_is_word:
                char[i] = ' ' + char[i]
            else:
                char[i] = char[i] + ' '
            pre_is_word = False
            pre_is_punct = True
        else:
            if pre_is_punct:
                char[i] = ' ' + char[i]
            pre_is_word = True
            pre_is_punct = False

    return ''.join(char)


def replace_multi_space(str):
    str = re.sub(' +', ' ', str)
    return str


def tokenize(input_str):
    input_str = re.sub('^,', '', input_str)
    input_str = re.sub('\.+\s*\.', '.', input_str) # replace multiple period with one period
    tokens = word_tokenize(input_str)
    for i in range(len(tokens)):
        tokens[i] = norm_word(tokens[i])
        # if i != len(tokens) - 1:
        #     tokens[i] = tokens[i].replace('.','')

    input_str = " ".join(tokens)
    input_str = remove_puntuation(input_str)
    input_str = revert_norm_word(input_str)

    return replace_multi_space(input_str)


def save_csv_v2(src, tgt, csv_file):
    assert len(src) == len(tgt), '[Warning] The training instance count is not equal.'
    csv_writer = csv.writer(open(csv_file, 'w'), delimiter='\t')
    csv_writer.writerow(['src', 'tgt'])
    for i in range(len(src)):
        csv_writer.writerow([src[i], tgt[i]])


def prepare_seq2seq(lines):
    src_all = list()
    tgt_all = list()
    for line in lines:
        line = tokenize(line)
        src = line.replace(',','')
        src = src.replace('.','')
        src = src.replace('?','')
        src = src.lower()
        src_all.append(src.strip())
        tgt_all.append(line.strip())

    return src_all, tgt_all


def read_lines(f_file):
    fo = open(f_file, 'r')
    return fo.readlines()


def read_csv(f_csv, cols):
    df = pd.read_csv(f_csv, delimiter='\t')
    lines = list()
    for i in range(len(cols)):
        lines.append(df[cols[i]].values.tolist())

    return lines


def norm_vnmese_accent(str):
    words = str.split(' ')
    for i in range(len(words)):
        if len(words[i]) <= 3:
            if not words[i].startswith('qu'):
                words[i] = words[i].replace("uỳ", "ùy")
                words[i] = words[i].replace("uý", "úy")
                words[i] = words[i].replace("uỷ", "ủy")
                words[i] = words[i].replace("uỹ", "ũy")
                words[i] = words[i].replace("uỵ", "ụy")
            else:
                words[i] = words[i].replace("ùy", "uỳ")
                words[i] = words[i].replace("úy", "uý")
                words[i] = words[i].replace("ủy", "uỷ")
                words[i] = words[i].replace("ũy", "uỹ")
                words[i] = words[i].replace("ụy", "uỵ")

            words[i] = words[i].replace("oà", "òa")
            words[i] = words[i].replace("oá", "óa")
            words[i] = words[i].replace("oả", "ỏa")
            words[i] = words[i].replace("oã", "õa")
            words[i] = words[i].replace("oạ", "ọa")
            words[i] = words[i].replace("oè", "òe")
            words[i] = words[i].replace("oé", "óe")
            words[i] = words[i].replace("oẻ", "ỏe")
            words[i] = words[i].replace("oẽ", "õe")
            words[i] = words[i].replace("oẹ", "ọe")
        else:
            words[i] = words[i].replace("òa", "oà")
            words[i] = words[i].replace("óa", "oá")
            words[i] = words[i].replace("ỏa", "oả")
            words[i] = words[i].replace("õa", "oã")
            words[i] = words[i].replace("ọa", "oạ")
            words[i] = words[i].replace("òe", "oè")
            words[i] = words[i].replace("óe", "oé")
            words[i] = words[i].replace("ỏe", "oẻ")
            words[i] = words[i].replace("õe", "oẽ")
            words[i] = words[i].replace("ọe", "oẹ")

    return ' '.join(words)


def load_lexicon(full_path, assert2fields=False, value_processor=None):
    phones = set()
    if value_processor is None:
        value_processor = lambda x: x[0]
    lex = {}
    for line in codecs.open(full_path, mode='r', encoding='utf-8'):
        line = norm_vnmese_accent(line)
        # line = unicodedata.normalize("NFC", line)
        parts = line.strip().split()
        if assert2fields:
            assert (len(parts) == 2)
        lex[parts[0]] = value_processor(parts[1:])
        for p in parts[1:]:
            phones.add(p)

    return lex, phones


def build_phone_vocab(vi_lexicon_file, foreign_lexicon_file):
    viLex, viPhones = load_lexicon(vi_lexicon_file, value_processor = lambda x: " ".join(x))
    foreignLex, foreignPhones = load_lexicon(foreign_lexicon_file, value_processor = lambda x: " ".join(x))

    phones = set(list(viPhones) + list(foreignPhones))
    vocab = list(phones)

    symbol_to_id = {s: i for i, s in enumerate(vocab)}
    id_to_symbol = {i: s for i, s in enumerate(vocab)}

    lexicon = (viLex, foreignLex)

    return symbol_to_id, id_to_symbol, lexicon

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'


def G2P(text, viLex, foreignLex):
    text = norm_vnmese_accent(text)
    # text = unicodedata.normalize("NFC", text)
    # chars = re.escape(string.punctuation)
    # line = text.strip().lower()
    # tmp = re.sub(' +',' ',line).split()
    tmp_string = ""

    for nword in text.split():
        if nword in viLex:  # found in vietnamese lexicon
            lexout = viLex[nword]
            tmp_string = tmp_string + ' ' + re.sub(' ', '|', ' '.join(lexout.split()))
        elif nword in foreignLex:
            lexout = foreignLex[nword]
            tmp_string = tmp_string + ' ' + re.sub(' ', '|', ' '.join(lexout.split()))
        else:
            if nword not in _punctuation:
                tmp_string = tmp_string + ' _'  # _ for UNK
            else:
                tmp_string = tmp_string + ' ' + nword

    # print(text)
    # print(tmp_string.strip())

    return tmp_string.strip()


def process_vin_list(f_vin, g2p_csv):
    symbol_to_id, id_to_symbol, lexicon = build_phone_vocab(
        'resources/all-vietnamese-syllables_17k9.XSAMPA.Mien-BAC.lex', 'resources/Foreign-Lexicon-6k.lex')
    fo = open(f_vin, 'r')
    f_error = open('resources/vin_errors.txt', 'w')
    df = pd.read_csv(g2p_csv, delimiter='\t')
    graphes_ = df.graph_word.values.tolist()
    phoneme_ = df.phoneme.values.tolist()
    g2p = dict(zip(graphes_, phoneme_))

    for line in fo:
        pairs = line.split('\t')
        graphes = pairs[0].lower()
        phonemes = pairs[1]
        graph_tokens = graphes.split()
        phoneme_tokens = phonemes.split()
        if len(graph_tokens) != len(phoneme_tokens):
            f_error.write(line)
        else:
            for i in range(len(graph_tokens)):
                phoneme_tokens[i] = phoneme_tokens[i].replace('_', ' ')
                phoneme_token_converted = G2P(phoneme_tokens[i], lexicon[0], lexicon[1])
                phoneme_token_converted = phoneme_token_converted.replace(' ', ' $ ').replace('|', ' | ')
                g2p[graph_tokens[i]] = phoneme_token_converted

    out_file = g2p_csv.replace('.csv', '_v2.csv')
    csv_writer = csv.writer(open(out_file, 'w'), delimiter='\t')
    csv_writer.writerow(['graph_word', 'graph_char', 'phoneme'])
    for key, value in g2p.items():
        csv_writer.writerow([key, ' '.join(list(str(key))), value])


if __name__ == '__main__':
    # make_vocab(param.source_train, param.src_vocab)
    # make_vocab(param.target_train, param.tgt_vocab)

    # graphes, phonemes = process_g2p_english('task_g2p_vnmese_syllable_sampa/dataset/all-vietnamese-syllables_17k9.XSAMPA.Mien-BAC.lex')
    # graphes, phonemes = process_g2p_vnmese('task_g2p_vnmese_withtone_localization_sampa_v2/dataset/Foreign-Lexicon-13k-6k-27k.lex')
    # save_list(graphes, 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/graph_vnmese_withtone_sampa.txt')
    # save_list(phonemes, 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/phonemes_vnmese_withtone_sampa.txt')
    # save_csv(graphes, phonemes, 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/g2p_vnmese_withtone_sampa.csv')
    # word2char('task_g2p_vnmese_withtone_localization_sampa_v2/dataset/graph_vnmese_withtone_sampa.txt', 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/graph_vnmese_withtone_sampa_char.txt')

    # lines = load_tone_data('corpora/addtone/train_tokenizer.csv')
    # lines = load_foreign_words('dataset/norm_foreign/foreign.csv')
    # lines, line_errors = load_foreign_words_v2('dataset/norm_foreign/foreign.csv')
    # save_file(line_errors, 'dataset/norm_foreign_v2/', 'line_errors.txt')
    # lines = load_seq2seq('dataset/auto_upper_punct_seq2seq/shard_0000_auto_upper_punct_seq2seq.csv', 'src', 'tgt')
    # lines = load_seq2seq('dataset/spoken2written/norm_seq2seq.csv', 'tag', 'written', 'spoken')
    # lines = load_g2p('dataset/g2p/foreign-lexicon-13k_one_type.lex')
    # lines = load_g2p_english('task_g2p_vnmese_withtone_localization_sampa_v2/dataset/graph_vnmese_withtone_sampa_char.txt', 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/phonemes_vnmese_withtone_sampa.txt')
    # split_train_test(lines, 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/')
    # read_data('dataset/norm_foreign/norm_foreign.pkl')

    # lines = read_csv('task_diacritics_restoration/dataset/train_tokenizer.csv', ['no_tone', 'tone'])
    # src_all, tgt_all = prepare_seq2seq(lines[1])
    # save_csv_v2(src_all, tgt_all, 'task_punct_restoration/dataset/dataset.csv')
    # lines = load_csv_2_cols('task_punct_restoration/dataset/dataset.csv', 'src', 'tgt')
    # split_train_test(lines, 'task_punct_restoration/dataset/', train_ratio=0.95, val_ratio=0.025)

    # lines = load_csv_2_cols('task_spoken2written/dataset/norm_seq2seq.csv', 'src', 'tgt')
    # lines = load_csv_3_cols('task_written2spoken/dataset/norm_seq2seq.csv', 'tag', 'src', 'tgt')
    # split_train_test(lines, 'task_written2spoken/dataset/', train_ratio=0.9, val_ratio=0.05)

    # convert graph2sampa
    # symbol_to_id, id_to_symbol, lexicon = build_phone_vocab('resources/all-vietnamese-syllables_17k9.XSAMPA.Mien-BAC.lex', 'resources/Foreign-Lexicon-6k.lex')
    # output = G2P('vin e co', lexicon[0], lexicon[1])
    # print(output)

    process_vin_list('resources/vinGroup_list.txt', 'task_g2p_vnmese_withtone_localization_sampa_v3/dataset/g2p_vnmese_withtone_sampa.csv')

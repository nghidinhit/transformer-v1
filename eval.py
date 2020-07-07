import os
import math
import ntpath
import codecs

import numpy as np

from hyperparams import Hyperparams as params
from data_load import load_test_data, load_source_vocab, load_target_vocab, convert_word2idx, load_vocab, load_data
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from transformer_v2 import Transformer
from torch.autograd import Variable
import torch
import utils
import time
import re
import copy


def eval():
    # Load data
    source_idxes, target_idxes, source_texts, target_texts = load_data(params.source_test, params.target_test, params.num_sample)
    source2idx, idx2source = load_vocab(params.src_vocab)
    target2idx, idx2target = load_vocab(params.tgt_vocab)
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    model.load_state_dict(torch.load(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '_best.pth'))
    print('Model Loaded.')
    model.eval()
    model.cuda()
    # Inference
    if not os.path.exists('results'):
        os.mkdir('results')
    print('len sources: ', len(source_idxes))
    print('batch size: ', params.batch_size)
    with open(params.eval_result + '/model%d.txt' % params.eval_epoch, 'w') as fout:
        list_of_refs, hypotheses = [], []
        list_of_refs_label, list_of_hypotheses_label = [], []
        scores = list()
        for i in range(len(source_idxes) // params.batch_size):
            # Get mini-batches
            source_idx_batches = source_idxes[i * params.batch_size : (i + 1) * params.batch_size]
            source_text_batches = source_texts[i * params.batch_size : (i + 1) * params.batch_size]
            target_text_batches = target_texts[i * params.batch_size : (i + 1) * params.batch_size]

            # Autoregressive inferencemultihead_attention
            source_idx_batches_tensor = Variable(torch.LongTensor(source_idx_batches).cuda())
            preds_t = torch.LongTensor(np.zeros((params.batch_size, params.maxlen), np.int32)).cuda()
            preds = Variable(preds_t)
            for j in range(params.maxlen):
                _, _preds, _ = model(source_idx_batches_tensor, preds)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())
            preds = preds.data.cpu().numpy()

            # Write to file
            for source, target, pred in zip(source_text_batches, target_text_batches, preds):  # sentence-wise
                got = " ".join(idx2target[idx] for idx in pred).split("</s>")[0].strip()
                target = target.replace(' | ', ' ').replace(' $ ', ' ')
                got = got.replace(' | ', ' ').replace(' $ ', ' ')
                # fout.write("-   source: " + source + "\n")
                # print("-   source: " + source)
                # fout.write("- expected: " + target + "\n")
                # print("- expected: " + target)
                # fout.write("-      got: " + got + "\n\n")
                # print("-      got: " + got + "\n")
                fout.write(source.replace(' ', '') + '\t' + got + '\n')
                fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()

                ref_label = utils.tag_sample(ref)
                hypothesis_label = utils.tag_sample(hypothesis)

                # ref_label_filter, hypothesis_label_filter = utils.filter_tag(ref, hypothesis, ref_label, hypothesis_label)

                if len(ref_label) > 1:
                    list_of_refs_label.extend(ref_label)
                    list_of_hypotheses_label.extend(hypothesis_label)

                # ref = target.replace(' | ', ' ').replace(' $ ', ' ').split()
                # hypothesis = got.replace(' | ', ' ').replace(' $ ', ' ').split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)

            # Calculate precision, recall, f1
            # pre, rec, f1 = utils.cal_tag_acc(list_of_refs_label, list_of_hypotheses_label)
            # print('pre: ', pre, ' - rec: ', rec, ' - f1: ', f1)
            # fout.write('pre: ' + str(pre) + ' - rec: ' + str(rec) + ' - f1: ' + str(f1))
            # Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            scores.append(score)
            # print('Bleu Score: ', score)
            # fout.write("Bleu Score = " + str(100 * score))
        # fout.write('\nBleu Score MEAN: ' + str(np.array(scores).mean()))
        # print('Bleu Score MEAN: ', np.array(scores).mean())


def upfirst(word):
    c = list(word)
    c[0] = c[0].upper()
    return ''.join(c)


def generate_from_label(source, label):
    source_token = source.split()
    label_token = label.split()
    if len(source_token) == len(label_token):
        for i in range(len(label_token)):
            if label_token[i] == 'B-UPALL-PERIOD':
                source_token[i] = source_token[i].upper() + '.'
            elif label_token[i] == 'B-UPALL-COMMA':
                source_token[i] = source_token[i].upper() + ','
            elif label_token[i] == 'B-UPALL-QUESTION':
                source_token[i] = source_token[i] + '?'
            elif label_token[i] == 'B-UPFIRST-PERIOD':
                source_token[i] = upfirst(source_token[i]) + '.'
            elif label_token[i] == 'B-UPFIRST-COMMA':
                source_token[i] = upfirst(source_token[i]) + '.'
            elif label_token[i] == 'B-UPFIRST-QUESTION':
                source_token[i] = upfirst(source_token[i]) + '?'
            elif label_token[i] == 'B-UPALL':
                source_token[i] = source_token[i].upper()
            elif label_token[i] == 'B-UPFIRST':
                source_token[i] = upfirst(source_token[i])
            elif label_token[i] == 'B-PERIOD':
                source_token[i] = source_token[i] + '.'
            elif label_token[i] == 'B-COMMA':
                source_token[i] = source_token[i] + ','
            elif label_token[i] == 'B-QUESTION':
                source_token[i] = source_token[i] + '?'

    else:
        print('-------------------------------------------------')
        print('len source != len(label)')
        print(source)
        print(label)

    return ' '.join(source_token)


def eval_v2():
    model, source2idx, idx2target = load_model(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '_best.pth')
    source_idxes, target_idxes, source_texts, target_texts = load_data(params.source_test, params.target_test, params.num_sample)
    list_of_refs_label, list_of_hypotheses_label = list(), list()
    list_of_refs, hypotheses = list(), list()
    print('num_sample: ', len(source_texts))
    count = 0
    fout = open(params.eval_result + '/model%d.txt' % params.eval_epoch, 'w')
    for source_text, target_text in zip(source_texts, target_texts):
        if len(source_text.split()) == len(target_text.split()):
            count += 1
            print('\rcount: %i' % count, end='\r')
            sample = source_text.lower().replace('.', '').replace(',', '').replace('?', '')
            # print("- source: " + sample.rstrip())
            fout.write('\n\n-----------------------------------------------------------------------\n')
            fout.write("- source: " + sample.rstrip() + '\n')
            # print("- target: " + target_text.rstrip())
            fout.write("- target: " + target_text.rstrip() + '\n')
            start_time = time.time()
            predict = infer(model, source2idx, idx2target, sample)
            end_time = time.time()
            time_infer = end_time - start_time
            # print("-    got: " + predict.rstrip())
            # predict_verify = verify_decode(sample, predict)
            predict_verify = predict
            time_verify = time.time() - end_time
            # gendata from label
            # predict = generate_from_label(sample, predict)
            #########################################################

            # print("- verify: " + predict_verify.rstrip())
            fout.write("-    got: " + predict.rstrip() + '\n')
            fout.write("- verify: " + predict_verify.rstrip() + '\n')
            if len(predict_verify.split()) != len(sample.split()):
                print('len(predict_verify) != len(source): ' + str(len(predict_verify)) + ' != ' + str(len(sample.split())))
                print("- source: " + sample.rstrip())
                print("- target: " + target_text.rstrip())
                fout.write('---------------------------------------------------------------' + '\n')

                fout.write('len(predict_verify) != len(source): ' + str(len(predict_verify)) + ' != ' + str(len(sample.split())) + '\n')
            fout.write('\ntime for infer: ' + str(time_infer) + '\n')
            # print('\ntime for infer: ', time_infer)
            fout.write('time for verify: ' + str(time_verify))
            # print('time for verify: ', time_verify)

            ref = target_text.split()
            hypothesis = predict_verify.split()

            ref_label = utils.tag_sample(ref)
            hypothesis_label = utils.tag_sample(hypothesis)

            if len(ref) != len(hypothesis) or len(ref_label) != len(hypothesis_label):
                print('**************************************')
                print(ref)
                print(hypothesis)
                print(ref_label)
                print(hypothesis_label)
                print('**************************************')

            if len(ref_label) > 1:
                list_of_refs_label.extend(ref_label)
                list_of_hypotheses_label.extend(hypothesis_label)

            # ref = target.replace(' | ', ' ').replace(' $ ', ' ').split()
            # hypothesis = got.replace(' | ', ' ').replace(' $ ', ' ').split()
            if len(ref) > 3 and len(hypothesis) > 3:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)

            if count % 100 == 0:
                # Calculate precision, recall, f1
                fout.write('\ncount sample = ' + str(count))
                pre, rec, f1 = utils.cal_tag_acc(list_of_refs_label, list_of_hypotheses_label)
                print('\npre: ', pre, ' - rec: ', rec, ' - f1: ', f1)
                fout.write('pre: ' + str(pre) + ' - rec: ' + str(rec) + ' - f1: ' + str(f1) + '\n')
                # Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                print('Bleu Score: ', score)
                fout.write("Bleu Score = " + str(100 * score) + '\n')


def eval_v3():
    model, source2idx, idx2target = load_model(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '_best.pth')
    source_idxes, target_idxes, source_texts, target_texts = load_data(params.source_test, params.target_test, params.num_sample)
    list_of_refs_label, list_of_hypotheses_label = list(), list()
    list_of_refs, hypotheses = list(), list()
    print('num_sample: ', len(source_texts))
    count = 0
    fout = open(params.eval_result + '/model%d.txt' % params.eval_epoch, 'w')
    for source_text, target_text in zip(source_texts, target_texts):
        count += 1
        print('\rcount: %i' % count, end='\r')
        sample = source_text
        # print("- source: " + sample.rstrip())
        fout.write('\n\n-----------------------------------------------------------------------\n')
        fout.write("- source: " + sample.rstrip() + '\n')
        # print("- target: " + target_text.rstrip())
        fout.write("- target: " + target_text.rstrip() + '\n')
        start_time = time.time()
        predict = infer(model, source2idx, idx2target, sample)
        end_time = time.time()
        time_infer = end_time - start_time
        # print("-    got: " + predict.rstrip())
        #########################################################

        # print("- verify: " + predict_verify.rstrip())
        fout.write("-    got: " + predict.rstrip() + '\n')
        fout.write('\ntime for infer: ' + str(time_infer) + '\n')
        # print('\ntime for infer: ', time_infer)

        ref = target_text.split()
        hypothesis = predict.split()


        if len(ref) > 3 and len(hypothesis) > 3:
            list_of_refs.append([ref])
            hypotheses.append(hypothesis)

        if count % 100 == 0:
            # Calculate precision, recall, f1
            fout.write('\ncount sample = ' + str(count))
            # Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            print('Bleu Score: ', score)
            fout.write("Bleu Score = " + str(100 * score) + '\n')


def verify_decode(source, predict):
    source_words = source.split()
    predict_words = predict.split()
    verify_words = copy.deepcopy(source_words)
    index_decode = 0
    for i in range(0, len(source_words)):
        if i < len(predict_words):
            if source_words[i] == predict_words[index_decode].lower().replace('.', '').replace(',', '').replace('?', ''):
                verify_words[i] = predict_words[index_decode]
                index_decode += 1
            elif source_words[i] == predict_words[i].lower().replace('.', '').replace(',', '').replace('?', ''):
                verify_words[i] = predict_words[i]
                index_decode = i + 1
    return ' '.join(verify_words)


def infer(model, source2idx, idx2target, sample):
    len_sample = len(sample.split())
    batch_size = 1
    sample_idx = convert_word2idx(source2idx, [sample])

    # Autoregressive inferencemultihead_attention
    start_time = time.time()
    source_idx_tensor = Variable(torch.LongTensor(sample_idx).cuda())
    preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32)).cuda()
    preds = Variable(preds_t)
    for j in range(len_sample):
        _, _preds, _ = model(source_idx_tensor, preds)
        got = " ".join(idx2target[idx] for idx in _preds.data.cpu().numpy()[0]).split("</s>")[0].strip()
        # print('got ' + str(j) + ': ', got)
        preds_t[:, j] = _preds.data[:, j]
        preds = Variable(preds_t.long())
    preds = preds.data.cpu().numpy()
    end_time = time.time()
    # print('infer: ', end_time - start_time)

    source_words = sample.split()
    pred_words = list()
    for i in range(len(source_words)):
        pred_word = idx2target[preds[0][i]]
        source_word = source_words[i]
        if pred_word == '<unk>':
            pred_words.append(source_word)
        elif pred_word == '</s>':
            break
        else:
            pred_words.append(pred_word)

    # print('check: ', time.time() - end_time)

    got = ' '.join(pred_words)

    # got = " ".join(idx2target[idx] for idx in preds[0]).split("</s>")[0].strip()
    # got = " ".join(idx2target[idx] for idx in preds[0]).split("</s>")[0].strip().replace(' | ', '|').replace(' $ ', ' ')
    # print("- source: " + sample)
    # print("-    got: " + got + "\n")
    return got


def load_model(model_path):
    # Load data
    source2idx, idx2source = load_vocab(params.src_vocab)
    target2idx, idx2target = load_vocab(params.tgt_vocab)
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    model.load_state_dict(torch.load(model_path))
    print('Model Loaded.')
    model.eval()
    model.cuda()

    return model, source2idx, idx2target


def eval_anhHuy(anh_huy):
    # source_idxes, target_idxes, source_texts, target_texts = load_data(params.source_test, params.target_test,
    #                                                                    params.num_sample)
    fin = open(anh_huy, 'r')
    preds = []
    source_texts = []
    target_texts = []
    for line in fin:
        source = line.split('\t')[0]
        source_texts.append(source)
        target = line.split('\t')[1]
        target_texts.append(target)
        pred = line.split('\t')[2]
        preds.append(pred)
        print('source: ', source)
        print('target: ', target)
        print('got: ', pred)

    list_of_refs, hypotheses = [], []
    list_of_refs_label, list_of_hypotheses_label = [], []
    scores = list()
    fout = open('task_g2p_anhKhoa_v2/dataset/result_en_vn_separated' + '/model%d_' % params.eval_epoch + ntpath.basename(anh_huy), 'w')
    for source, target, pred in zip(source_texts, target_texts, preds):  # sentence-wise
        got = pred
        target = target.replace(' | ', ' ').replace(' $ ', ' ')
        got = got.replace(' | ', ' ').replace(' $ ', ' ')
        fout.write("-   source: " + source + "\n")
        # print("-   source: " + source)
        fout.write("- expected: " + target + "\n")
        # print("- expected: " + target)
        fout.write("-      got: " + got + "\n")
        # print("-      got: " + got + "\n")
        fout.flush()

        # bleu score
        ref = target.split()
        hypothesis = got.split()

        ref_label = utils.tag_sample(ref)
        hypothesis_label = utils.tag_sample(hypothesis)

        # ref_label_filter, hypothesis_label_filter = utils.filter_tag(ref, hypothesis, ref_label, hypothesis_label)

        if len(ref_label) > 1:
            list_of_refs_label.extend(ref_label)
            list_of_hypotheses_label.extend(hypothesis_label)

        # if len(ref) > 3 and len(hypothesis) > 3:
        list_of_refs.append([ref])
        hypotheses.append(hypothesis)
        print('list_of_refs: ', list_of_refs)
        print('hypotheses: ', hypotheses)
        score = corpus_bleu(list_of_refs, hypotheses)
        scores.append(score)
        # print('Bleu Score: ', score)
        # fout.write("Bleu Score = " + str(100 * score))
    fout.write('\nBleu Score MEAN: ' + str(np.array(scores).mean()))
    print('Bleu Score MEAN: ', np.array(scores).mean())


if __name__ == '__main__':
    # eval()
    # eval_anhHuy('task_g2p_anhKhoa_v2/dataset/Huy/00_10kForeignWords.lex.done')
    # eval_anhHuy('task_g2p_anhKhoa_v2/dataset/Huy/01_10k_27k_ForeignWords.lex.done')
    # eval_anhHuy('task_g2p_anhKhoa_v2/dataset/Huy/02_10kForeign_27kForeign_17kVnSylable.lex.done')
    # eval_anhHuy('task_g2p_anhKhoa_v2/dataset/Huy/03_10kForeign_27kForeign_125kEnglish.lex.done')
    # eval_anhHuy('task_g2p_anhKhoa_v2/dataset/Huy/04_10kForeign_27kForeign_125kEnglish_17kVNSylable.lex.done')
    eval_anhHuy('task_g2p_anhKhoa_v2/dataset/result_en_vn_separated/en_04_nGram.txt')
    # eval_v3()
    print('Done')
    #
    # model, source2idx, idx2target = load_model(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '_best.pth')
    # while True:
    #     sample = input('Input your sentence: ')
    #     # sample = ' '.join(list(sample))
    #     output = infer(model, source2idx, idx2target, sample)
    #     print(output)




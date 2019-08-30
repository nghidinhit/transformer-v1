import codecs
import os

import numpy as np

from hyperparams import Hyperparams as params
from data_load import load_test_data, load_source_vocab, load_target_vocab, convert_word2idx
from nltk.translate.bleu_score import corpus_bleu
from transformer import Transformer
from torch.autograd import Variable
import torch


def eval():
    # Load data
    source_idxes, source_texts, target_texts = load_test_data()
    source2idx, idx2source = load_source_vocab()
    target2idx, idx2target = load_target_vocab()
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    model.load_state_dict(torch.load(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '.pth'))
    print('Model Loaded.')
    model.eval()
    model.cuda()
    # Inference
    if not os.path.exists('results'):
        os.mkdir('results')
    with open('results/model%d.txt' % params.eval_epoch, 'w') as fout:
        list_of_refs, hypotheses = [], []
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
                print(pred)
                print(idx2target)
                got = " ".join(idx2target[idx] for idx in pred).split("</s>")[0].strip()
                # fout.write("-   source: " + source + "\n")
                print("-   source: " + source)
                # fout.write("- expected: " + target + "\n")
                print("- expected: " + target)
                # fout.write("-      got: " + got + "\n\n")
                print("-      got: " + got + "\n")
                # fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)
            # Calculate bleu score
            # score = corpus_bleu(list_of_refs, hypotheses)
            # fout.write("Bleu Score = " + str(100 * score))
            break


def infer(model, source2idx, idx2target, sample):
    batch_size = 1
    sample_idx = convert_word2idx(source2idx, [sample])

    # Autoregressive inferencemultihead_attention
    source_idx_tensor = Variable(torch.LongTensor(sample_idx).cuda())
    preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32)).cuda()
    preds = Variable(preds_t)
    for j in range(params.maxlen):
        _, _preds, _ = model(source_idx_tensor, preds)
        preds_t[:, j] = _preds.data[:, j]
        preds = Variable(preds_t.long())
    preds = preds.data.cpu().numpy()

    got = " ".join(idx2target[idx] for idx in preds[0]).split("</s>")[0].strip()
    print("- source: " + sample)
    print("-    got: " + got + "\n")


def load_model(model_path):
    # Load data
    source2idx, idx2source = load_source_vocab()
    target2idx, idx2target = load_target_vocab()
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    model.load_state_dict(torch.load(model_path))
    print('Model Loaded.')
    model.eval()
    model.cuda()

    return model, source2idx, idx2target


if __name__ == '__main__':
    # eval()
    print('Done')

    model, source2idx, idx2target = load_model(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '.pth')
    while True:
        sample = input('Input your sentence: ')
        sample = ' '.join(list(sample))
        infer(model, source2idx, idx2target, sample)




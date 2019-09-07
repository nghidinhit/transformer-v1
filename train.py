
from hyperparams import Hyperparams as params
from data_load import get_batch_indices, load_source_vocab, load_target_vocab, load_vocab, load_data

from transformer import Transformer
from torch.autograd import Variable
from data_load import load_train_data
from tensorboardX import SummaryWriter
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pickle as pickle
from tqdm import tqdm


def train():
    current_batches = 0
    src2idx, idx2src = load_vocab(params.src_vocab)
    tgt2idx, idx2tgt = load_vocab(params.tgt_vocab)
    encoder_vocab = len(src2idx)
    decoder_vocab = len(tgt2idx)
    writer = SummaryWriter()
    # Load data
    source_idxes, target_idxes = load_data(params.source_train, params.target_train)
    # calc total batch False
    num_batch = len(source_idxes) // params.batch_size
    model = Transformer(params, encoder_vocab, decoder_vocab)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.train()
    if params.use_gpu:
        model.cuda()
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(params.model_dir):
        os.makedirs(params.model_dir)
    # if params.preload is not None and os.path.exists(params.model_dir + '/history.pkl'):
    #     with open(params.model_dir + '/history.pkl') as in_file:
    #         history = pickle.load(in_file)
    # else:
    #     history = {'current_batches': 0}
    # current_batches = history['current_batches']
    if params.preload is not None:
        model.load_state_dict(torch.load(params.preload))

    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=[0.9, 0.98], eps=1e-8)
    # if params.preload is not None and os.path.exists(params.model_dir + '/optimizer.pth'):
    #     optimizer.load_state_dict(torch.load(params.model_dir + '/optimizer.pth'))

    # if params.preload is not None and os.path.exists(params.model_dir + '/model_epoch_%02d.pth' % params.preload):
    #     model.load_state_dict(torch.load(params.model_dir + '/model_epoch_%02d.pth' % params.preload))
    # startepoch = int(params.preload) if params.preload is not None else 1
    startepoch = 6

    for epoch in range(startepoch, params.num_epochs + 1):
        current_batch = 0
        for index, current_index in tqdm(get_batch_indices(len(source_idxes), params.batch_size)):
            tic = time.time()
            if params.use_gpu:
                x_batch = Variable(torch.LongTensor(source_idxes[index]).cuda())
                y_batch = Variable(torch.LongTensor(target_idxes[index]).cuda())
            else:
                x_batch = Variable(torch.LongTensor(source_idxes[index]))
                y_batch = Variable(torch.LongTensor(target_idxes[index]))
            toc = time.time()
            tic_r = time.time()
            # torch.cuda.synchronize()
            optimizer.zero_grad()
            loss, _, acc = model(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
            toc_r = time.time()
            current_batches += 1
            current_batch += 1
            if current_batches % 10 == 0:
                writer.add_scalar('./loss', loss.data.cpu().numpy(), current_batches)
                writer.add_scalar('./acc', acc.data.cpu().numpy(), current_batches)
            if current_batches % 5 == 0:
                print('\r\t\t\t  epoch %d - batch %d/%d - loss %f - acc %f' % (epoch, current_batch, num_batch, loss.item(), acc.item()), end='\r')
                # print('\rbatch loading used time %f, model forward used time %f' % (toc - tic, toc_r - tic_r), end='\r')
            if current_batches % 100 == 0:
                writer.export_scalars_to_json(params.model_dir + '/all_scalars.json')
        # with open(hp.model_dir + '/history.pkl', 'w') as out_file:
        #     pickle.dump(history, out_file)
        checkpoint_path = params.model_dir + '/model_epoch_%02d' % epoch + '.pth'
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), params.model_dir + '/optimizer.pth')


if __name__ == '__main__':
    train()

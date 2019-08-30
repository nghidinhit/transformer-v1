
class Hyperparams:
    '''Hyperparameters'''

    source_train = 'corpora/g2p_english/train.char.src.txt'
    target_train = 'corpora/g2p_english/train.tgt.txt'
    source_test = 'corpora/g2p_english/test.char.src.txt'
    target_test = 'corpora/g2p_english/test.tgt.txt'

    src_vocab = 'g2p_english.src.vocab.tsv'
    tgt_vocab = 'g2p_english.tgt.vocab.tsv'
    model_dir = './models/g2p_english'

    g2p_english = 'corpora/g2p_english/g2p_english.txt'
    graph_english = 'corpora/g2p_english/graph_english.txt'
    graph_english_char = 'corpora/g2p_english/graph_english_char.txt'
    phoneme_english = 'corpora/g2p_english/phoneme_english.txt'

    # training
    batch_size = 64 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory


    # model
    maxlen = 30 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 100  # epoch of model for eval
    preload = None  # epcho of preloaded model for resuming training
    '''======================================================================================================='''
    # data
    # source_train = 'corpora/de-en/train.tags.de-en.de'
    # target_train = 'corpora/de-en/train.tags.de-en.en'
    # source_test = 'corpora/de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    # target_test = 'corpora/de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    # src_vocab = 'de.vocab.tsv'
    # tgt_vocab = 'en.vocab.tsv'
    #
    # # training
    # batch_size = 32  # alias = N
    # lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'logdir'  # log directory
    #
    # model_dir = './models/'  # saving directory
    #
    # # model
    # maxlen = 10  # Maximum number of words in a sentence. alias = T.
    # # Feel free to increase this if you are ambitious.
    # min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512  # alias = C
    # num_blocks = 6  # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 20  # epoch of model for eval
    # preload = None  # epcho of preloaded model for resuming training

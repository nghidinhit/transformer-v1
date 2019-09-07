
class Hyperparams:
    '''Hyperparameters'''

    ''' >100k english sampa format'''
    # source_train = 'corpora/g2p_english/train.char.src.txt'
    # target_train = 'corpora/g2p_english/train.tgt.txt'
    # source_test = 'corpora/g2p_english/test.char.src.txt'
    # target_test = 'corpora/g2p_english/test.tgt.txt'
    #
    # src_vocab = 'g2p_english.src.vocab.tsv'
    # tgt_vocab = 'g2p_english.tgt.vocab.tsv'
    # model_dir = './models/g2p_english'
    #
    # g2p_english = 'corpora/g2p_english/g2p_english.txt'
    # graph_english = 'corpora/g2p_english/graph_english.txt'
    # graph_english_char = 'corpora/g2p_english/graph_english_char.txt'
    # phoneme_english = 'corpora/g2p_english/phoneme_english.txt'

    ''' 25k vnmese sampa format'''
    # source_train = 'corpora/g2p_vnmese_samba/train.src.txt'
    # target_train = 'corpora/g2p_vnmese_samba/train.tgt.txt'
    # source_test = 'corpora/g2p_vnmese_samba/test.src.txt'
    # target_test = 'corpora/g2p_vnmese_samba/test.tgt.txt'
    #
    # src_vocab = 'g2p_vnmese_samba.src.vocab.tsv'
    # tgt_vocab = 'g2p_vnmese_samba.tgt.vocab.tsv'
    # model_dir = './models/g2p_vnmese_samba'

    ''' 20k vnmese + 20k english sampa format'''
    # source_train = 'corpora/g2p_vnmese_english_samba/train_vnmese_english_40k.src.txt'
    # target_train = 'corpora/g2p_vnmese_english_samba/train_vnmese_english_40k.tgt.txt'
    # source_test = 'corpora/g2p_vnmese_english_samba/test_vnmese.src.txt'
    # target_test = 'corpora/g2p_vnmese_english_samba/test_vnmese.tgt.txt'
    #
    # src_vocab = 'g2p_vnmese_english_samba_40k.src.vocab.tsv'
    # tgt_vocab = 'g2p_vnmese_english_samba_40k.tgt.vocab.tsv'
    # model_dir = './models/g2p_vnmese_english_samba_40k'

    ''' 20k vnmese + >100k english sampa format'''
    # source_train = 'corpora/g2p_vnmese_english_samba/train_vnmese_english.src.txt'
    # target_train = 'corpora/g2p_vnmese_english_samba/train_vnmese_english.tgt.txt'
    # source_test = 'corpora/g2p_vnmese_english_samba/test_vnmese.src.txt'
    # target_test = 'corpora/g2p_vnmese_english_samba/test_vnmese.tgt.txt'
    #
    # src_vocab = 'g2p_vnmese_english_samba.src.vocab.tsv'
    # tgt_vocab = 'g2p_vnmese_english_samba.tgt.vocab.tsv'
    # model_dir = './models/g2p_vnmese_english_samba'

    ''' 20k vnmese + 17k syllable sampa format'''
    # source_train = 'task_g2p_vnmese_withtone_localization_syllable_sampa/dataset/train_val.src.txt'
    # target_train = 'task_g2p_vnmese_withtone_localization_syllable_sampa/dataset/train_val.tgt.txt'
    # source_test = 'task_g2p_vnmese_withtone_localization_syllable_sampa/dataset/test.src.txt'
    # target_test = 'task_g2p_vnmese_withtone_localization_syllable_sampa/dataset/test.tgt.txt'
    #
    # src_vocab = 'task_g2p_vnmese_withtone_localization_syllable_sampa/vocab/g2p_vnmese_withtone_localizatin_syllable.src.vocab.tsv'
    # tgt_vocab = 'task_g2p_vnmese_withtone_localization_syllable_sampa/vocab/g2p_vnmese_withtone_localizatin_syllable.tgt.vocab.tsv'
    # model_dir = 'task_g2p_vnmese_withtone_localization_syllable_sampa/model/task_g2p_vnmese_withtone_localization_syllable_sampa'
    # eval_result = 'task_g2p_vnmese_withtone_localization_syllable_sampa/result'

    ''' 20k vnmese v2'''
    # source_train = 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/train_val.src.txt'
    # target_train = 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/train_val.tgt.txt'
    # source_test = 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/test.src.txt'
    # target_test = 'task_g2p_vnmese_withtone_localization_sampa_v2/dataset/test.tgt.txt'
    #
    # src_vocab = 'task_g2p_vnmese_withtone_localization_sampa_v2/vocab/g2p_vnmese_withtone_localizatin_syllable.src.vocab.tsv'
    # tgt_vocab = 'task_g2p_vnmese_withtone_localization_sampa_v2/vocab/g2p_vnmese_withtone_localizatin_syllable.tgt.vocab.tsv'
    # model_dir = 'task_g2p_vnmese_withtone_localization_sampa_v2/model/task_g2p_vnmese_withtone_localization_syllable_sampa'
    # eval_result = 'task_g2p_vnmese_withtone_localization_sampa_v2/result'
    #
    ''' 20k vnmese v3'''
    # source_train = 'task_g2p_vnmese_withtone_localization_sampa_v3/dataset/train_val.src.txt'
    # target_train = 'task_g2p_vnmese_withtone_localization_sampa_v3/dataset/train_val.tgt.txt'
    # source_test = 'task_g2p_vnmese_withtone_localization_sampa_v3/dataset/test.src.txt'
    # target_test = 'task_g2p_vnmese_withtone_localization_sampa_v3/dataset/test.tgt.txt'
    #
    # src_vocab = 'task_g2p_vnmese_withtone_localization_sampa_v3/vocab/g2p_vnmese_withtone_localizatin.src.vocab.tsv'
    # tgt_vocab = 'task_g2p_vnmese_withtone_localization_sampa_v3/vocab/g2p_vnmese_withtone_localizatin.tgt.vocab.tsv'
    # model_dir = 'task_g2p_vnmese_withtone_localization_sampa_v3/model/task_g2p_vnmese_withtone_localization_sampa'
    # eval_result = 'task_g2p_vnmese_withtone_localization_sampa_v3/result'
    #
    # # training
    # batch_size = 64 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'logdir' # log directory
    #
    # # model
    # maxlen = 30 # Maximum number of words in a sentence. alias = T.
    #             # Feel free to increase this if you are ambitious.
    # min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512 # alias = C
    # num_blocks = 6 # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 100  # epoch of model for eval
    # preload = None


    ''' task punct restoration '''
    # is_lower = False
    # source_train = 'task_punct_restoration/dataset/train.src.txt'
    # target_train = 'task_punct_restoration/dataset/train.tgt.txt'
    # source_test = 'task_punct_restoration/dataset/test.src.txt'
    # target_test = 'task_punct_restoration/dataset/test.tgt.txt'
    #
    # src_vocab = 'task_punct_restoration/vocab/punct_restoration.src.vocab.tsv'
    # tgt_vocab = 'task_punct_restoration/vocab/punct_restoration.tgt.vocab.tsv'
    # model_dir = 'task_punct_restoration/model'
    # eval_result = 'task_punct_restoration/result'
    #
    # # training
    # batch_size = 8 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'logdir' # log directory
    #
    # # model
    # maxlen = 100 # Maximum number of words in a sentence. alias = T.
    #             # Feel free to increase this if you are ambitious.
    # min_cnt = 15 # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512 # alias = C
    # num_blocks = 6 # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 4  # epoch of model for eval
    # preload = 'task_punct_restoration/model/model_epoch_05.pth'


    ''' task written2spoken'''
    is_lower = True
    source_train = 'task_written2spoken/dataset/train_val.src.txt'
    target_train = 'task_written2spoken/dataset/train_val.tgt.txt'
    source_test = 'task_written2spoken/dataset/test.src.txt'
    target_test = 'task_written2spoken/dataset/test.tgt.txt'

    src_vocab = 'task_written2spoken/vocab/task_written2spoken.src.vocab.tsv'
    tgt_vocab = 'task_written2spoken/vocab/task_written2spoken.tgt.vocab.tsv'
    model_dir = 'task_written2spoken/model'
    eval_result = 'task_written2spoken/result'

    # training
    batch_size = 8 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory

    # model
    maxlen = 100 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 4  # epoch of model for eval
    # preload = 'task_written2spoken/model/model_epoch_05.pth'
    preload = None
    use_gpu = False

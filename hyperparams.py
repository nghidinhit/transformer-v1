
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
    # num_sample = 1000000000
    # source_train = 'task_punct_restoration/dataset/train.src.txt'
    # target_train = 'task_punct_restoration/dataset/train.tgt.txt'
    # target_train_label = 'task_punct_restoration/dataset/train.label.tgt.txt'
    # source_test = 'task_punct_restoration/dataset/test.src.txt'
    # target_test = 'task_punct_restoration/dataset/test.tgt.txt'
    # target_test_label = 'task_punct_restoration/dataset/test.label.tgt.txt'
    # source_val = 'task_punct_restoration/dataset/val.src.txt'
    # target_val = 'task_punct_restoration/dataset/val.tgt.txt'
    # target_val_label = 'task_punct_restoration/dataset/val.label.tgt.txt'
    #
    # src_vocab = 'task_punct_restoration/vocab/punct_restoration.src.vocab.tsv'
    # tgt_vocab = 'task_punct_restoration/vocab/punct_restoration.tgt.vocab.tsv'
    # model_dir = 'task_punct_restoration/model'
    # eval_result = 'task_punct_restoration/result'
    #
    # # training
    # batch_size = 8 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'task_punct_restoration/logdir' # log directory
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
    # eval_epoch = 21  # epoch of model for eval
    # preload = 'task_punct_restoration/model/model_epoch_21.pth'
    # use_gpu = True


    ''' task punct restoration with label '''
    # is_lower = False
    # num_sample = 10000000
    # source_train = 'task_punct_restoration/dataset/train.filter.src.txt'
    # # target_train = 'task_punct_restoration/dataset/train.filter.tgt.txt'
    # target_train = 'task_punct_restoration/dataset/train.label.tgt.txt'
    # source_test = 'task_punct_restoration/dataset/test.filter.src.txt'
    # # target_test = 'task_punct_restoration/dataset/test.filter.tgt.txt'
    # target_test = 'task_punct_restoration/dataset/test.label.tgt.txt'
    # source_val = 'task_punct_restoration/dataset/val.filter.src.txt'
    # # target_val = 'task_punct_restoration/dataset/val.filter.tgt.txt'
    # target_val = 'task_punct_restoration/dataset/val.label.tgt.txt'
    #
    # # src_vocab = 'task_punct_restoration/vocab/punct_restoration.filter.src.vocab.tsv'
    # # tgt_vocab = 'task_punct_restoration/vocab/punct_restoration.filter.tgt.vocab.tsv'
    # src_vocab = 'task_punct_restoration/vocab/punct_restoration.label.src.vocab.tsv'
    # tgt_vocab = 'task_punct_restoration/vocab/punct_restoration.label.tgt.vocab.tsv'
    # model_dir = 'task_punct_restoration/model-filter'
    # eval_result = 'task_punct_restoration/result-filter'
    #
    # # training
    # batch_size = 32 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'task_punct_restoration/logdir' # log directory
    #
    # # model
    # maxlen = 100 # Maximum number of words in a sentence. alias = T.
    #             # Feel free to increase this if you are ambitious.
    # min_cnt = 15 # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512 # alias = C
    # num_blocks = 6 # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.3
    # sinusoid = False # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 11  # epoch of model for eval
    # preload = 'task_punct_restoration/model/model_epoch_22.pth'
    # use_gpu = True

    ''' task punct restoration with label and word'''
    # is_lower = False
    # num_sample = 10000000
    # source_train = 'task_punct_restoration/dataset/train.filter.src.txt'
    # target_train_word = 'task_punct_restoration/dataset/train.filter.tgt.txt'
    # target_train_label = 'task_punct_restoration/dataset/train.label.tgt.txt'
    # source_test = 'task_punct_restoration/dataset/test.filter.src.txt'
    # target_test_word = 'task_punct_restoration/dataset/test.filter.tgt.txt'
    # target_test_label = 'task_punct_restoration/dataset/test.label.tgt.txt'
    # source_val = 'task_punct_restoration/dataset/val.filter.src.txt'
    # target_val_word = 'task_punct_restoration/dataset/val.filter.tgt.txt'
    # target_val_label = 'task_punct_restoration/dataset/val.label.tgt.txt'
    #
    # src_vocab_word = 'task_punct_restoration/vocab/punct_restoration.filter.src.vocab.tsv'
    # tgt_vocab_word = 'task_punct_restoration/vocab/punct_restoration.filter.tgt.vocab.tsv'
    # src_vocab_label = 'task_punct_restoration/vocab/punct_restoration.label.src.vocab.tsv'
    # tgt_vocab_label = 'task_punct_restoration/vocab/punct_restoration.label.tgt.vocab.tsv'
    # src_vocab_word_label = 'task_punct_restoration/vocab/punct_restoration.word_label.src.vocab.tsv'
    # tgt_vocab_word_label = 'task_punct_restoration/vocab/punct_restoration.word_label.tgt.vocab.tsv'
    # model_dir = 'task_punct_restoration/model_word_label'
    # eval_result = 'task_punct_restoration/result_word_label'
    #
    # # training
    # batch_size = 16  # alias = N
    # lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'task_punct_restoration/logdir_word_label'  # log directory
    #
    # # model
    # maxlen = 100  # Maximum number of words in a sentence. alias = T.
    # # Feel free to increase this if you are ambitious.
    # min_cnt = 15  # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512  # alias = C
    # num_blocks = 6  # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 11  # epoch of model for eval
    # preload = None
    # use_gpu = True

    ''' task written2spoken'''
    # is_lower = True
    # num_sample = 1000
    # source_train = 'task_written2spoken/dataset/train_val.src.txt'
    # target_train = 'task_written2spoken/dataset/train_val.tgt.txt'
    # source_test = 'task_written2spoken/dataset/test.src.txt'
    # target_test = 'task_written2spoken/dataset/test.tgt.txt'
    #
    # src_vocab = 'task_written2spoken/vocab/task_written2spoken.src.vocab.tsv'
    # tgt_vocab = 'task_written2spoken/vocab/task_written2spoken.tgt.vocab.tsv'
    # model_dir = 'task_written2spoken/model'
    # eval_result = 'task_written2spoken/result'
    #
    # # training
    # batch_size = 8 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'logdir' # log directory
    #
    # # model
    # maxlen = 100 # Maximum number of words in a sentence. alias = T.
    #             # Feel free to increase this if you are ambitious.
    # min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512 # alias = C
    # num_blocks = 6 # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 4  # epoch of model for eval
    # # preload = 'task_written2spoken/model/model_epoch_05.pth'
    # preload = None
    # use_gpu = False

    ''' task_spoken2written'''
    # is_lower = True
    # num_sample = 1000000
    # source_train = 'task_spoken2written/dataset/train_val.src.txt'
    # target_train = 'task_spoken2written/dataset/train_val_char.tgt.txt'
    # source_test = 'task_spoken2written/dataset/test.src.txt'
    # target_test = 'task_spoken2written/dataset/test_char.tgt.txt'
    #
    # src_vocab = 'task_spoken2written/vocab/task_spoken2written.src.vocab.tsv'
    # tgt_vocab = 'task_spoken2written/vocab/task_spoken2written_char.tgt.vocab.tsv'
    # model_dir = 'task_spoken2written/model'
    # eval_result = 'task_spoken2written/result'
    #
    # # training
    # batch_size = 32 # alias = N
    # lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    # logdir = 'task_spoken2written/logdir' # log directory
    #
    # # model
    # maxlen = 100 # Maximum number of words in a sentence. alias = T.
    #             # Feel free to increase this if you are ambitious.
    # min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    # hidden_units = 512 # alias = C
    # num_blocks = 6 # number of encoder/decoder blocks
    # num_epochs = 100
    # num_heads = 8
    # dropout_rate = 0.1
    # sinusoid = False # If True, use sinusoid. If false, positional embedding.
    # eval_epoch = 21  # epoch of model for eval
    # preload = 'task_spoken2written/model/model_epoch_22_best.pth'
    # # preload = None
    # use_gpu = True

    ''' g2p anh Khoa'''
    experiment_num = 'ex4'
    filename_test = '05_test_100_new_foreign_words.lex'
    is_lower = False
    num_sample = 1000000
    source_train = 'task_g2p_anhKhoa_v2/dataset/' + experiment_num + '/train_val.src.txt'
    target_train = 'task_g2p_anhKhoa_v2/dataset/' + experiment_num + '/train_val.tgt.txt'
    # source_test = 'task_g2p_anhKhoa/dataset/ex1/test.src.txt'
    # target_test = 'task_g2p_anhKhoa/dataset/ex1/test.tgt.txt'
    source_test = 'task_g2p_anhKhoa_v2/dataset/' + experiment_num + '/' + filename_test + '_graph.txt'
    target_test = 'task_g2p_anhKhoa_v2/dataset/' + experiment_num + '/' + filename_test + '_phonemes.txt'

    src_vocab = 'task_g2p_anhKhoa_v2/vocab/' + experiment_num + '/task_g2p_anhKhoa_v2.src.vocab.tsv'
    tgt_vocab = 'task_g2p_anhKhoa_v2/vocab/' + experiment_num + '/task_g2p_anhKhoa_v2.tgt.vocab.tsv'
    model_dir = 'task_g2p_anhKhoa_v2/model/' + experiment_num
    eval_result = 'task_g2p_anhKhoa_v2/result/' + experiment_num

    # training
    start_epoch = 0
    batch_size = 90  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'task_g2p_anhKhoa_v2/logdir/' + experiment_num  # log directory

    # model
    maxlen = 30  # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 100  # epoch of model for eval
    # preload = 'task_spoken2written/model/model_epoch_22_best.pth'
    preload = None
    use_gpu = True

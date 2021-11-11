class DefaultConfigs:
    config_path = 'config.py'

    ############################################
    # modify this part to set up the experiments
    lg1 = 'en'
    lg2 = 'fr'
    train_data_path = "paracrawl/enfr.test" # find this in the "data.zip" file
    # due to size limitation, we only uploaded first 300,000 lines of English-French for your checking
    validation_size = 1000 # for the training described in the paper, we used 10000 for validation. Here we use 1000 sentences for test.
    ############################################

    output_dir = "ckpt"
    best_models =  "best_model/"
    pooling_method = 'MEAN'
    ffn_act = 'swish'
    hiddent_act = 'swish'
    bpe_path = 'preprocessing/' + lg1 + lg2 + '_bpe_50000.model'
    lower = True
    vocab_size = 50000
    
    # hyperparameters
    max_seq_len = 200
    train_batch_size = 128     
    vali_batch_size = 128
    num_layers = 2
    sent_dim = 512
    token_dim = 512
    num_heads = 8
    ffn_dim = 1024
    neg_samples = 4
    mlmrate = 0.15
    sentence_alignment_loss_weight = 2
    sentence_similarity_loss_weight = 2
    
    weight_decay = 1e-5
    learning_rate = 1e-3
    lr_decay_from = 3 # lr decay from which epoch
    if lg2 == 'it':
        lr_decay_from = 6
    num_train_epochs = 12
    if lg2 == 'it':
        num_train_epochs = 30

    has_validation = True
    
    # sentence evaluation part
    eval_data_path = 'data/'
    eval_batch_size = 256
    n_keys = 200000
    n_queries = 2000
    method = 'nn'        
   
    def set_config(self, model_name, resume=False, is_train=True):
        if is_train:
            self.emb_dropout = 0.1
            self.attention_dropout = 0.1
            self.hidden_dropout = 0.1
        else:
            self.emb_dropout = 0.0
            self.attention_dropout = 0.0
            self.hidden_dropout = 0.0
            self.bpe_path = '../preprocessing/enfr_bpe_50000.model'
            self.output_dir = "../" + output_dir
        self.resume = resume
        self.model_name = model_name
        
        # experimental settings for UGT and UGT+ALIGN+SIM
        # other settings can be added here either for hyperparameter search and ablation study
        if model_name =='UGT':
            self.has_FC = True
            self.sentloss_before_fc = True
            self.has_sentence_loss = False
            self.has_sentence_similarity_loss = False
        elif model_name =='UGT+ALIGN+SIM':
            self.has_FC = True
            self.sentloss_before_fc = True
            self.has_sentence_loss = True
            self.has_sentence_similarity_loss = True            
        else:
            print('No such experiments! Please check.')
            raise

config = DefaultConfigs()


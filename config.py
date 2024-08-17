import configparser


class Node_Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print('loading config: %s failed!' % (config_file))

        # self.seed = conf.getint('Model_Setup', 'seed')
        self.epoch = conf.getint('Model_Setup', 'epoch')
        self.lr = conf.getfloat('Model_Setup', 'lr')
        self.weight_decay = conf.getfloat('Model_Setup', 'weight_decay')
        self.hidden_dim = conf.getint('Model_Setup', 'hidden_dim')
        self.eval_indicators = conf['Model_Setup']['eval_indicators'].split(',')
        self.num_layers = conf.getint('Model_Setup', 'num_layers')
        self.note = conf['Model_Setup']['note']
        self.model = conf['Model_Setup']['model']

        """数据集设置"""
        self.dataset_name = conf['Data_Setting']['dataset_name']
        self.num_features = conf.getint('Data_Setting', 'num_features')
        self.num_task = conf.getint('Data_Setting', 'num_task')
        self.task = conf['Data_Setting']['task']

        """Transformer设置"""
        self.num_heads = conf.getint('Transformer_Setup', 'num_heads')
        self.lpapas_ratio = conf.getfloat('Transformer_Setup','lpapas_ratio')
        self.qkv_bias = conf.getboolean('Transformer_Setup','qkv_bias')


class Graph_Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print('loading config: %s failed!' % (config_file))

        """模型设置"""
        self.seed = conf.getint('Model_Setup', 'seed')
        self.epoch = conf.getint('Model_Setup', 'epoch')
        self.lr = conf.getfloat('Model_Setup', 'lr')
        self.weight_decay = conf.getfloat('Model_Setup', 'weight_decay')
        self.hidden_dim = conf.getint('Model_Setup', 'hidden_dim')
        self.batch_size = conf.getint('Model_Setup', 'batch_size')
        self.fold = conf.getint('Model_Setup', 'fold')
        self.eval_indicators = conf['Model_Setup']['eval_indicators'].split(',')
        self.dropout = conf.getfloat('Model_Setup', 'dropout')


        """卷积设置"""
        self.embeding_method = conf['Embed_Setting']['embeding_method']
        self.embeding_layer = conf.getint('Embed_Setting', 'embeding_layer')

        """池化设置"""

        self.pool_key = conf.getboolean('Pool_Setting', 'pool_key')
        self.pool_layer = conf.getint('Pool_Setting', 'pool_layer')
        self.pool_conv_method = conf['Pool_Setting']['pool_conv_method']
        self.pool_transformer_heads = conf.getint('Pool_Setting', 'pool_transformer_heads')
        self.lpapas_ratio = conf.getfloat('Pool_Setting', 'lpapas_ratio')
        self.qkv_bias = conf.getboolean('Pool_Setting', 'qkv_bias')
        self.global_topy = conf.getboolean('Pool_Setting', 'global_topy')
        self.local_topy = conf.getboolean('Pool_Setting', 'local_topy')

        self.aggration_method = conf['Pool_Setting']['aggration_method']
        self.trans_layer = conf.getint('Pool_Setting', 'trans_layer')
        self.local_pool_layer = conf.getint('Pool_Setting', 'local_pool_layer')



        """数据集设置"""
        self.dataset_name = conf['Data_Setting']['dataset_name']
        self.num_features = conf.getint('Data_Setting', 'num_features')
        self.num_task = conf.getint('Data_Setting', 'num_task')
        # Data setting
        self.num_class = conf.getint('Data_Setting', 'num_class')  # number of graph type
        self.feat_dim = conf.getint('Data_Setting', 'feat_dim')  # feature dimension
        self.input_dim = conf.getint('Data_Setting', 'input_dim')  # input dimension
        self.attr_dim = conf.getint('Data_Setting', 'attr_dim')  # attribute dimension
        self.test_number = conf.getint('Data_Setting',
                                       'test_number')  # if specified, will overwrite -fold and use the last -test_number graphs as testing data

        self.note = conf['Model_Setup']['note']
        self.seed = conf.getint('Model_Setup', 'seed')
        self.hidden_dim = conf.getint('Model_Setup', 'hidden_dim')
        self.gcn_layer = conf.getint('Model_Setup', 'gcn_layer')
        self.gin_layer = conf.getint('Model_Setup', 'gin_layer')
        self.dropout = conf.getfloat('Model_Setup', 'dropout')
        self.epochs = conf.getint('Model_Setup', 'epochs')
        self.lr = conf.getfloat('Model_Setup', 'lr')
        self.weight_decay = conf.getfloat('Model_Setup', 'weight_decay')
        self.batch_size = conf.getint('Model_Setup', 'batch_size')
        self.readout = conf['Model_Setup']['readout']
        self.fold = conf.getint('Model_Setup', 'fold')  # (1...10) fold cross validation
        self.model = conf['Model_Setup']['model']
        self.optimizer = conf['Model_Setup']['optimizer']

        #GCN seetings
        self.gcn_res = conf.getint('GCN_Setup', 'gcn_res') # whether to normalize gcn layers
        self.gcn_norm = conf.getint('GCN_Setup', 'gcn_norm') # whether to normalize gcn layers/
        self.bn = conf.getint('GCN_Setup', 'bn') # whether to normalize gcn layers
        self.relu = conf['GCN_Setup']['relu'] # whether to use relu

        #Pool seetings
        self.percent = conf.getfloat('Pool_Setting', 'percent') # agcn node keep percent(=k/node_num)
        self.hierarchical_num = conf.getint('Pool_Setting', 'hierarchical_num') # pooling layer number

        #Muti-channel settings
        self.local_topology = conf.getboolean('Muti_channel_Setting', 'local_topology')
        self.global_topology = conf.getboolean('Muti_channel_Setting', 'global_topology')
        self.with_feature = conf.getboolean('Muti_channel_Setting', 'with_feature')
        self.Channel_3 = conf['Muti_channel_Setting']['Channel_3']
        self.Channel_2 = conf['Muti_channel_Setting']['Channel_2']
        self.Channel_1 = conf['Muti_channel_Setting']['Channel_1']

        #Feature Channel settings
        self.transformer_layer_number = conf.getint('Feature_Channel_Setup', 'transformer_layer_number')
        self.transformer_dropout = conf.getfloat('Feature_Channel_Setup', 'transformer_dropout')
        self.transformer_head_number = conf.getint('Feature_Channel_Setup', 'transformer_head_number')
        self.transformer_norm_input = conf.getboolean('Feature_Channel_Setup', 'transformer_norm_input')
        self.transformer_forward = conf.getint('Feature_Channel_Setup', 'transformer_forward')
        self.bilstm_layer_number = conf.getint('Feature_Channel_Setup', 'bilstm_layer_number')
        self.bilstm_drop_out = conf.getfloat('Feature_Channel_Setup', 'bilstm_drop_out')
        self.bilstm_bias = conf.getboolean('Feature_Channel_Setup', 'bilstm_bias')
        self.gru_layer_number = conf.getint('Feature_Channel_Setup', 'gru_layer_number')
        self.gru_drop_out = conf.getfloat('Feature_Channel_Setup', 'gru_drop_out')
        self.gru_bias = conf.getboolean('Feature_Channel_Setup', 'gru_bias')
        self.get_att_score = conf['Feature_Channel_Setup']['get_att_score']
        self.att_norm_X = conf.getboolean('Feature_Channel_Setup', 'att_norm_X')
        self.trans_X_3 = conf.getboolean('Feature_Channel_Setup', 'trans_X_3')
        self.aggregation_3 = conf.getfloat('Feature_Channel_Setup', 'aggregation_3')


        # Local topology Channel Setting
        self.trans_X_1 = conf.getboolean('Local_topology_Channel_Setup', 'trans_X_1')
        self.aggregation_1 = conf.getfloat('Local_topology_Channel_Setup', 'aggregation_1')


        # DiffPool Setting
        self.diffPool_max_num_nodes = conf.getint('DiffPool_Setting', 'diffPool_max_num_nodes')
        self.diffPool_num_gcn_layer = conf.getint('DiffPool_Setting', 'diffPool_num_gcn_layer')
        self.diffPool_assign_ratio = conf.getfloat('DiffPool_Setting', 'diffPool_assign_ratio')
        self.diffPool_num_classes = conf.getint('DiffPool_Setting', 'diffPool_num_classes')
        self.diffPool_num_pool = conf.getint('DiffPool_Setting', 'diffPool_num_pool')
        self.diffPool_bn = conf.getboolean('DiffPool_Setting', 'diffPool_bn')
        self.diffPool_bias = conf.getboolean('DiffPool_Setting', 'diffPool_bias')
        self.diffPool_dropout = conf.getfloat('DiffPool_Setting', 'diffPool_dropout')


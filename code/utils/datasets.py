from pathlib import Path

class l3das22:
    ''' L3DAS22 dataset

    '''
    def __init__(self, root_dir, cfg):
        self.root_dir = Path(root_dir).joinpath(cfg['dataset'])
        self.dataset_dir = dict()
        self.dataset_dir['task2'] = {
            'dev' : self.root_dir.joinpath('L3DAS22_Task2_dev').joinpath('L3DAS22_Task2_dev'),
            'train' : self.root_dir.joinpath('L3DAS22_Task2_train').joinpath('L3DAS22_Task2_train'),
            'test' : self.root_dir.joinpath('L3DAS22_Task2_test').joinpath('L3DAS22_Task2_test')
        }
        self.label_set_task2 = ['Computer_keyboard', 'Drawer_open_or_close', 'Cupboard_open_or_close', \
            'Finger_snapping', 'Keys_jangling', 'Knock', 'Laughter', 'Scissors', 'Telephone', \
                'Writing', 'Chink_and_clink', 'Printer', 'Female_speech_and_woman_speaking', 'Male_speech_and_man_speaking']
        self.label_dic_task2 = {'Chink_and_clink':0,
                                'Computer_keyboard':1,
                                'Cupboard_open_or_close':2,
                                'Drawer_open_or_close':3,
                                'Female_speech_and_woman_speaking':4,
                                'Finger_snapping':5,
                                'Keys_jangling':6,
                                'Knock':7,
                                'Laughter':8,
                                'Male_speech_and_man_speaking':9,
                                'Printer':10,
                                'Scissors':11,
                                'Telephone':12,
                                'Writing':13}

        self.clip_length = 30 # seconds long
        self.label_resolution = 0.1 # 0.1s is the label resolution
        self.ov_str_index = -6 # string index indicating overlap
        self.split_str_index = 5 # string index indicating split number
        self.format_str_index = -1 # string index indicating ambisonics number
        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_set_task2)
        if cfg['data']['label_normalize'] == 'respective':
            self.max_loc_value = [2, 1.5, 1] 
        elif cfg['data']['label_normalize'] == 'unified' :
            self.max_loc_value = [2, 2, 2]
        else:
            self.max_loc_value = [1, 1, 1]

class l3das21:
    ''' L3DAS21 dataset

    '''
    def __init__(self, root_dir, cfg):
        self.root_dir = Path(root_dir).joinpath(cfg['dataset'])
        self.dataset_dir = dict()
        self.dataset_dir['task2'] = {
            'dev' : self.root_dir.joinpath('Task2').joinpath('L3DAS_Task2_dev'),
            'train' : self.root_dir.joinpath('Task2').joinpath('L3DAS_Task2_train'),
            'test' : self.root_dir.joinpath('Task2').joinpath('L3DAS_Task2_test')
        }
        self.label_set_task2 = ['Computer_keyboard', 'Drawer_open_or_close', 'Cupboard_open_or_close', \
            'Finger_snapping', 'Keys_jangling', 'Knock', 'Laughter', 'Scissors', 'Telephone', \
                'Writing', 'Chink_and_clink', 'Printer', 'Female_speech_and_woman_speaking', 'Male_speech_and_man_speaking']
        self.label_dic_task2 = {'Chink_and_clink':0,
                                'Computer_keyboard':1,
                                'Cupboard_open_or_close':2,
                                'Drawer_open_or_close':3,
                                'Female_speech_and_woman_speaking':4,
                                'Finger_snapping':5,
                                'Keys_jangling':6,
                                'Knock':7,
                                'Laughter':8,
                                'Male_speech_and_man_speaking':9,
                                'Printer':10,
                                'Scissors':11,
                                'Telephone':12,
                                'Writing':13}

        self.clip_length = 60 # seconds long
        self.label_resolution = 0.1 # 0.1s is the label resolution
        self.ov_str_index = -6 # string index indicating overlap
        self.split_str_index = 5 # string index indicating split number
        self.format_str_index = -1 # string index indicating ambisonics number
        self.max_ov = 3 # max overlap
        self.num_classes = len(self.label_set_task2)

        # normalization of doa label
        if cfg['data']['label_normalize'] == 'respective':
            self.max_loc_value = [2, 1.5, 1] 
        elif cfg['data']['label_normalize'] == 'unified' :
            self.max_loc_value = [2, 2, 2]
        else:
            self.max_loc_value = [1, 1, 1]


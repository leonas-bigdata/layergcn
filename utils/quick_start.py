# coding: utf-8

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform, os

#============Newly added functions
def load_adjacency_list_to_df(file_path, uid_field, iid_field):
    records = []
    import pandas as pd
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line :
                continue
            tokens = line.split()
            user_id = int(tokens[0]) 
            for item_id in tokens[1:]:
                records.append({uid_field: user_id, iid_field: int(item_id)})
    return pd.DataFrame(records)

#===================================
def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    # dataset = RecDataset(config)
    # # print dataset statistics
    # logger.info(str(dataset))

    # train_dataset, valid_dataset, test_dataset = dataset.split(config['split_ratio'])
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']

    train_path = os.path.join('data', dataset, 'train.txt')
    test_path = os.path.join('data', dataset, 'test.txt')

    # Đọc file với đường dẫn đã ghép
    train_df = load_adjacency_list_to_df(train_path, uid_field, iid_field)
    test_df = load_adjacency_list_to_df(test_path, uid_field, iid_field)
    
    # Khởi tạo trực tiếp bằng tham số df (Skip hoàn toàn k-core và splitting)
    train_dataset = RecDataset(config, df=train_df)
    test_dataset = RecDataset(config, df=test_df)
    
    # Với các framework GCN, tập validation thường được lấy luôn là tập test 
    # (hoặc bạn có thể tự chia thêm validation df nếu muốn)
    # valid_dataset = RecDataset(config, df=test_df.copy())

    logger.info('\n====Training====\n' + str(train_dataset))
    # logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    # (valid_data, test_data) = (
    #     EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
    #     EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    (test_data) = (EvalDataLoader(config,  test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model Chỗ này để chạy model tạm thời comment để test code
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # random seed reset
        init_seed(config['seed'])

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        if idx==0:
            logger.info(model)
        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        # best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=save_model, is_test=True, idx=idx)
        #########
        # hyper_ret.append((hyper_tuple, best_valid_result, test_result))
        hyper_ret.append((hyper_tuple, test_result))

        # save best test
        if test_result[val_metric] > best_test_value:
            best_test_value = test_result[val_metric]
            best_test_idx = idx
        idx += 1

        # logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        # logger.info('test result: {}'.format(dict2str(test_result)))
        # logger.info('████Current BEST████:\nParameters: {}={},\n'
        #             'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
        #     hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

        logger.info('test result: {}'.format(dict2str(test_result)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Test: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, v) in hyper_ret: 
        logger.info('Parameters: {}={},\n best test: {}'.format(config['hyper_parameters'], p, dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1])))


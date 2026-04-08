# # coding: utf-8

# """
# Run application
# ##########################
# """
# from logging import getLogger
# from itertools import product
# from utils.dataset import RecDataset
# from utils.dataloader import TrainDataLoader, EvalDataLoader
# from utils.logger import init_logger
# from utils.configurator import Config
# from utils.utils import init_seed, get_model, get_trainer, dict2str
# import platform, os

# #============Newly added functions
# def load_adjacency_list_to_df(file_path, uid_field, iid_field):
#     records = []
#     import pandas as pd
#     with open(file_path, "r", encoding='utf-8') as fin:
#         for line in fin:
#             line = line.strip()
#             if not line :
#                 continue
#             tokens = line.split()
#             user_id = int(tokens[0]) 
#             for item_id in tokens[1:]:
#                 records.append({uid_field: user_id, iid_field: int(item_id)})
#     return pd.DataFrame(records)

# #===================================
# def quick_start(model, dataset, config_dict, save_model=True):
#     import pandas as pd
#     # merge config dict
#     config = Config(model, dataset, config_dict)
#     init_logger(config)
#     logger = getLogger()
#     # print config infor
#     logger.info('██Server: \t' + platform.node())
#     logger.info('██Dir: \t' + os.getcwd() + '\n')
#     logger.info(config)

#     # load data
#     # dataset = RecDataset(config)
#     # # print dataset statistics
#     # logger.info(str(dataset))

#     # train_dataset, valid_dataset, test_dataset = dataset.split(config['split_ratio'])
#     uid_field = config['USER_ID_FIELD']
#     iid_field = config['ITEM_ID_FIELD']

#     train_path = os.path.join('data', dataset, 'train.txt')
#     test_path = os.path.join('data', dataset, 'test.txt')

#     # Đọc file với đường dẫn đã ghép
#     train_df = load_adjacency_list_to_df(train_path, uid_field, iid_field)
#     test_df = load_adjacency_list_to_df(test_path, uid_field, iid_field)

    
    
#     # Khởi tạo trực tiếp bằng tham số df (Skip hoàn toàn k-core và splitting)
#     train_dataset = RecDataset(config, df=train_df)
#     test_dataset = RecDataset(config, df=test_df)
    
#     # Với các framework GCN, tập validation thường được lấy luôn là tập test 
#     # (hoặc bạn có thể tự chia thêm validation df nếu muốn)
#     # valid_dataset = RecDataset(config, df=test_df.copy())

#     logger.info('\n====Training====\n' + str(train_dataset))
#     # logger.info('\n====Validation====\n' + str(valid_dataset))
#     logger.info('\n====Testing====\n' + str(test_dataset))

#     # wrap into dataloader
#     train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
#     # (valid_data, test_data) = (
#     #     EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
#     #     EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

#     (test_data) = (EvalDataLoader(config,  test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

#     ############ Dataset loadded, run model Chỗ này để chạy model tạm thời comment để test code
#     hyper_ret = []
#     val_metric = config['valid_metric'].lower()
#     best_test_value = 0.0
#     idx = best_test_idx = 0

#     logger.info('\n\n=================================\n\n')

#     # hyper-parameters
#     hyper_ls = []
#     if "seed" not in config['hyper_parameters']:
#         config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
#     for i in config['hyper_parameters']:
#         hyper_ls.append(config[i] or [None])
#     # combinations
#     combinators = list(product(*hyper_ls))
#     total_loops = len(combinators)
#     for hyper_tuple in combinators:
#         for j, k in zip(config['hyper_parameters'], hyper_tuple):
#             config[j] = k

#         logger.info('========={}/{}: Parameters:{}={}======='.format(
#             idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

#         # random seed reset
#         init_seed(config['seed'])

#         # set random state of dataloader
#         train_data.pretrain_setup()
#         # model loading and initialization
#         model = get_model(config['model'])(config, train_data).to(config['device'])
#         if idx==0:
#             logger.info(model)
#         # trainer loading and initialization
#         trainer = get_trainer()(config, model)
#         # debug
#         # model training
#         # best_valid_score, best_valid_result = trainer.fit(train_data, test_data=test_data, saved=save_model)
#         trainer.fit(train_data, valid_data=test_data, test_data=test_data, saved=save_model)
#         # model evaluation
#         test_result = trainer.evaluate(test_data, load_best_model=save_model, is_test=True, idx=idx)
#         #########
#         # hyper_ret.append((hyper_tuple, best_valid_result, test_result))
#         hyper_ret.append((hyper_tuple, test_result))

#         # save best test
#         if test_result[val_metric] > best_test_value:
#             best_test_value = test_result[val_metric]
#             best_test_idx = idx
#         idx += 1

#         # logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
#         # logger.info('test result: {}'.format(dict2str(test_result)))
#         # logger.info('████Current BEST████:\nParameters: {}={},\n'
#         #             'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
#         #     hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

#         logger.info('test result: {}'.format(dict2str(test_result)))
#         logger.info('████Current BEST████:\nParameters: {}={},\n'
#                     'Test: {}\n\n\n'.format(config['hyper_parameters'],
#             hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1])))

#     # log info
#     logger.info('\n============All Over=====================')
#     for (p, v) in hyper_ret: 
#         logger.info('Parameters: {}={},\n best test: {}'.format(config['hyper_parameters'], p, dict2str(v)))

#     logger.info('\n\n█████████████ BEST ████████████████')
#     logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
#         config['hyper_parameters'],
#         hyper_ret[best_test_idx][0],
#         dict2str(hyper_ret[best_test_idx][1])))


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
import pandas as pd
import numpy as np


def load_adjacency_list_to_df(file_path, uid_field, iid_field):
    """Load adjacency list format: each line is `user_id item_id item_id ...`"""
    records = []
    with open(file_path, "r", encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            user_id = int(tokens[0])
            for item_id in tokens[1:]:
                records.append({uid_field: user_id, iid_field: int(item_id)})
    return pd.DataFrame(records)


def remap_ids(train_df, test_df, uid_field, iid_field):
    """
    Remap user/item IDs to contiguous 0-based integers using TRAIN set as reference.

    BUG ROOT CAUSE:
    - LayerGCN builds embedding matrices of size n_users x latent_dim and n_items x latent_dim
    - n_users/n_items come from dataset.num() = count of unique IDs
    - But interaction_matrix is built using the RAW integer IDs as row/col indices
    - If raw IDs are e.g. [0, 5, 10], embedding size = 3 but matrix uses index 10 → CRASH or silent wrong lookup
    - Even if IDs happen to be 0-based, train and test must share the SAME mapping
    - This causes embeddings to be indexed incorrectly → loss never decreases

    FIX: Remap to contiguous 0-based IDs from train, filter test to only known users/items.
    """
    # Build mapping from train only (model only knows train entities)
    uni_users = sorted(train_df[uid_field].unique())
    uni_items = sorted(train_df[iid_field].unique())

    u_map = {old: new for new, old in enumerate(uni_users)}
    i_map = {old: new for new, old in enumerate(uni_items)}

    # Remap train
    train_df = train_df.copy()
    train_df[uid_field] = train_df[uid_field].map(u_map)
    train_df[iid_field] = train_df[iid_field].map(i_map)

    # Remap test — drop rows with unseen users or items (cold-start)
    test_df = test_df.copy()
    test_df[uid_field] = test_df[uid_field].map(u_map)
    test_df[iid_field] = test_df[iid_field].map(i_map)
    before = len(test_df)
    test_df.dropna(inplace=True)
    test_df = test_df.astype(int)
    after = len(test_df)
    if before - after > 0:
        print(f"[remap_ids] Dropped {before - after} test interactions with unseen users/items (cold-start).")

    # Reset index
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    n_users = len(u_map)
    n_items = len(i_map)
    print(f"[remap_ids] n_users={n_users}, n_items={n_items}, "
          f"train_inters={len(train_df)}, test_inters={len(test_df)}")

    return train_df, test_df, n_users, n_items


def patch_dataset_num(dataset, n_users, n_items, uid_field, iid_field):
    """
    Monkey-patch dataset.num() to return the correct global count from train mapping.

    WHY: dataset.num(uid_field) calls len(pd.unique(self.df[uid_field])).
    For the TEST dataset this returns the number of unique users in TEST, not TRAIN.
    But LayerGCN uses train_data.dataset.num() to size its embedding matrices.
    If test has fewer users than train, the model would be built with wrong size.
    We pin the correct values here so both train/test datasets report consistent sizes.
    """
    _n_users = n_users
    _n_items = n_items
    _uid = uid_field
    _iid = iid_field
    original_num = dataset.num.__func__  # unbound

    def patched_num(self, field):
        if field == _uid:
            return _n_users
        if field == _iid:
            return _n_items
        return original_num(self, field)

    import types
    dataset.num = types.MethodType(patched_num, dataset)

    # Also patch inter_num, user_num, item_num used in __str__ and sparsity calc
    dataset.inter_num = len(dataset.df)
    dataset.user_num = n_users
    dataset.item_num = n_items


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']

    train_path = os.path.join('data', dataset, 'train.txt')
    test_path = os.path.join('data', dataset, 'test.txt')

    logger.info(f"Loading train from: {train_path}")
    logger.info(f"Loading test  from: {test_path}")

    # Load raw adjacency lists
    train_df_raw = load_adjacency_list_to_df(train_path, uid_field, iid_field)
    test_df_raw  = load_adjacency_list_to_df(test_path,  uid_field, iid_field)

    # ══════════════════════════════════════════════════════════════
    # CRITICAL FIX: Remap IDs to contiguous 0-based integers
    # Without this, interaction_matrix uses raw IDs as indices but
    # embedding tables are sized by unique-count → silent mismatch
    # → gradients flow but update WRONG embedding rows → loss stuck
    # ══════════════════════════════════════════════════════════════
    train_df, test_df, n_users, n_items = remap_ids(
        train_df_raw, test_df_raw, uid_field, iid_field)

    # Build datasets (bypasses k-core & splitting — data already prepared)
    train_dataset = RecDataset(config, df=train_df)
    test_dataset  = RecDataset(config, df=test_df)

    # Pin correct global sizes so model builds embedding tables correctly
    patch_dataset_num(train_dataset, n_users, n_items, uid_field, iid_field)
    patch_dataset_num(test_dataset,  n_users, n_items, uid_field, iid_field)

    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Testing====\n'  + str(test_dataset))

    # Wrap into dataloaders
    train_data = TrainDataLoader(
        config, train_dataset,
        batch_size=config['train_batch_size'], shuffle=True)
    test_data = EvalDataLoader(
        config, test_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'])

    # ── Hyper-parameter search loop ──────────────────────────────
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])

    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)

    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        init_seed(config['seed'])
        train_data.pretrain_setup()

        model_obj = get_model(config['model'])(config, train_data).to(config['device'])
        if idx == 0:
            logger.info(model_obj)

        trainer = get_trainer()(config, model_obj)
        trainer.fit(train_data, valid_data=test_data, test_data=test_data, saved=save_model)

        test_result = trainer.evaluate(
            test_data, load_best_model=save_model, is_test=True, idx=idx)

        hyper_ret.append((hyper_tuple, test_result))

        if test_result[val_metric] > best_test_value:
            best_test_value = test_result[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('test result: {}'.format(dict2str(test_result)))
        logger.info('████Current BEST████:\nParameters: {}={},\nTest: {}\n\n\n'.format(
            config['hyper_parameters'],
            hyper_ret[best_test_idx][0],
            dict2str(hyper_ret[best_test_idx][1])))

    logger.info('\n============All Over=====================')
    for (p, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best test: {}'.format(
            config['hyper_parameters'], p, dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(
        config['hyper_parameters'],
        hyper_ret[best_test_idx][0],
        dict2str(hyper_ret[best_test_idx][1])))


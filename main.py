# coding: utf-8
# UPDATE:

"""
Main entry
##########################
# HEADER = "user_id:token\titem_id:token\n"
# def conver_to_inter(src_path : str, dest_path: str):
#     total = 0
#     with open(src_path, "r", encoding = "utf-8") as fin, open (dest_path, "w", encoding = "utf-8") as fout:
#         fout.write(HEADER)
#         for line in fin:
#             line = line.strip()
#             if not line:
#                 continue
#             tokens = line.split()
#             user_id = tokens[0]
#             for item_id in tokens[1:]:
#                 fout.write(f"{user_id}\t{item_id}\n")
#                 total += 1
#     print(f' Written {total:, } interactions - > {dest_path}')

#     return total
"""


import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LayerGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)



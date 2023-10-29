import os
import argparse
from trainer import Trainer
from utils import setup_seed, my_makedir
from config import get_config
import torch
from data_loader import data_loader


def main(args):
    torch.backends.cudnn.benchmark = True
    setup_seed(args.seed)

    if args.mode == 'train':
        # load dataset
        clean_loader, train_loader, val_loader = data_loader(args.data_path, args.clean_data_num, args.train_data_num, args.batch_size, args.val_data_num, args.val_batch_size, args.num_workers)
        # training
        trainer = Trainer(clean_loader, train_loader, val_loader, args)
        trainer.train()
    else:
        raise NotImplementedError('Mode [{}] is not found'.format(args.mode))


if __name__ == '__main__':

    args = get_config()

    # if args.is_print_network:
    #     print(args)

    main(args)
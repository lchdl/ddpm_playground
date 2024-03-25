import argparse
from diffusion.utilities.database import Database
from diffusion.model_training.ddpm import DDPM_Loader, DDPM_Trainer
from diffusion.utilities.misc import hash_dict
from diffusion.utilities.file_ops import file_exist

def collect_train_data_FFHQ():
    #
    train_xlsx = 'ffhq_train.xlsx'
    val_xlsx = 'ffhq_val.xlsx'
    test_xlsx = None
    #
    ffhq_xlsx = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/FFHQ_thumbnails_128x128.xlsx'
    ffhq_db = Database(db_keys=DDPM_Loader.db_keys, xlsx_file=ffhq_xlsx)
    ffhq_db.shuffle(9669)
    train_ratio = 0.9
    ffhq_train, ffhq_val = ffhq_db.split([train_ratio, 1.0-train_ratio])
    ffhq_train.export_xlsx(train_xlsx)
    ffhq_val.export_xlsx(val_xlsx)
    #
    return train_xlsx, val_xlsx, test_xlsx

def collect_train_data_CelebA_debug():
    #
    train_xlsx = 'celeba_train_debug.xlsx'
    val_xlsx = 'celeba_val_debug.xlsx'
    test_xlsx = None
    xlsx = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/CelebA_cropped_128x128_debug.xlsx'
    db = Database(db_keys=DDPM_Loader.db_keys, xlsx_file=xlsx)
    db.shuffle(9669)
    train_ratio = 0.5
    db_train, db_val = db.split([train_ratio, 1.0-train_ratio])
    db_train.export_xlsx(train_xlsx)
    db_val.export_xlsx(val_xlsx)
    return train_xlsx, val_xlsx, test_xlsx


def generate_folder_from_model_configs(model_configs):
    return 'models/ddpm_%s/' % hash_dict(model_configs, 8)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_index', help='Run training on which GPU?', 
                        nargs='?', type=int, default=0)
    args = parser.parse_args()

    # set model and training configs
    model_configs={
        'in_ch': 3,
        'out_ch': 3,
        'base_ch': 128,
        'ch_mult': [1, 2, 4, 8], 
        'down_arch': ['rr', 'rar', 'rara', 'r'],
        'bottle_arch': 'arara',
        'up_arch': ['rr', 'rar', 'rara', 'r'],
        'temb_dims': [1024, 512],
        'tag_dim': 512,
        'num_groups': 2,
        'dropout_p': None,
    }
    train_configs={
        'model_folder': generate_folder_from_model_configs(model_configs) + '/celeba/',
        'use_cache': False,
        'apply_train_DA': True,
        'apply_val_DA': False,
        'apply_test_DA': False,
        'initial_lr': 2e-4,
        'epochs': 1000,
        'train_batch_size': 25,
        'val_batch_size': 25,
        'test_batch_size': 25,
        'save_every_n_epochs': None,
        'dataloader_workers': 16,
        'gpu_index': args.gpu_index,
    }

    train_xlsx, val_xlsx, test_xlsx = collect_train_data_CelebA_debug()

    # firing up the training process    
    trainer = DDPM_Trainer(train_configs, model_configs, 
                           train_xlsx, val_xlsx, test_xlsx)
    trainer.train(train_configs['epochs'], print_mode='best')

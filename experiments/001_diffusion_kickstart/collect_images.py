from diffusion.utilities.database import Database
from diffusion.utilities.file_ops import ls, gn, mkdir, join_path, file_exist
from diffusion.model_training.ddpm import DDPM_Loader
import imageio as iio
import numpy as np
from scipy.ndimage import zoom

def collect_ffhq_thumbnail_128x128():
    database = Database(db_keys=DDPM_Loader.db_keys)
    database_xlsx = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/FFHQ_thumbnails_128x128.xlsx'
    image_root_dir = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/ffhq_thumbnails_128x128/'
    all_image_files = [item for item in ls(image_root_dir, full_path=True) if item.endswith('.png')]
    for image_file in all_image_files:
        print(image_file)
        record = database.make_empty_record()
        record['Image Source'] = 'FFHQ_thumbnail'
        record['Image Name'] = gn(image_file, no_extension=True)
        record['Image Path'] = image_file
        record['Image Tags'] = 'ffhq;thumbnail;'
        record['Image Description'] = ''
        database.add_record(record)
    database.export_xlsx(database_xlsx, up_freezed_rows=1, left_freezed_cols=2)

def collect_celeba_218x178():
    database_xlsx = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/CelebA_cropped_128x128.xlsx'
    image_root_dir = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/img_align_celeba_png/'
    image_out_dir = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/img_align_celeba_128x128/'
    image_tag_txt = '/data7/chenghaoliu/Codes/DIP/diffusion/diffusion/datasets/celeba_attr_txt/list_attr_celeba.txt'

    image_attr = {
        'init': False,
        'all_tags': [
            # tag1, tag2, ...
        ],
        'tags_for_each_image':{
            # img_name: [tag1, tag2, ...]
        },
    }

    def _resize_image_to_128x128(x: np.ndarray, order=3, return_type='uint8'):
        # x: [h, w, 3]
        ch_r, ch_g, ch_b, w = x[:,:,0], x[:,:,1], x[:,:,2], x.shape[1]
        scale = 128 / w
        ch_r = zoom(ch_r, scale, order=order)
        ch_g = zoom(ch_g, scale, order=order)
        ch_b = zoom(ch_b, scale, order=order)
        h0 = ch_r.shape[0]//2
        ch_r = ch_r[h0-64:h0+64,0:128]
        ch_g = ch_g[h0-64:h0+64,0:128]
        ch_b = ch_b[h0-64:h0+64,0:128]
        x0 = np.zeros([128,128,3])
        x0[:,:,0] = ch_r
        x0[:,:,1] = ch_g
        x0[:,:,2] = ch_b
        x0 = x0.astype(return_type)
        return x0
    def _process_celeba_img(image_in, image_out):
        iio.imsave(image_out, _resize_image_to_128x128(iio.imread(image_in)))
    def _find_img_tag(image_file):
        def _read_image_attr():
            nonlocal image_attr
            with open(image_tag_txt, 'r') as fobj:
                for lineno, linebuf in enumerate(fobj): # readline()
                    if lineno==0: continue
                    elif lineno==1: image_attr['all_tags']=linebuf.strip().split(' ')
                    else:
                        words=[word for word in linebuf.strip().split(' ') if word != '']
                        name=words[0].replace('.jpg', '')
                        tags=[]
                        for labid, label in enumerate(words[1:]):
                            if label=='1': tags.append(image_attr['all_tags'][labid])
                        image_attr['tags_for_each_image'][name]=tags
            image_attr['init']=True
        # lazy init
        if not image_attr['init']: 
            _read_image_attr()
            print('all_tags:', image_attr['all_tags'])
        return image_attr['tags_for_each_image'][gn(image_file, no_extension=True)]

    database = Database(db_keys=DDPM_Loader.db_keys)
    mkdir(image_out_dir)
    all_image_files = [item for item in ls(image_root_dir, full_path=True) if item.endswith('.png')]
    for i, image_file in enumerate(all_image_files):
        print('[%d/%d] %s' % (i+1, len(all_image_files), gn(image_file)))
        image_out = join_path(image_out_dir, gn(image_file))
        if not file_exist(image_out):
            _process_celeba_img(image_file, image_out)
        record = database.make_empty_record()
        record['Image Source'] = 'CelebA'
        record['Image Name'] = gn(image_out, no_extension=True)
        record['Image Path'] = image_out
        record['Image Tags'] = ''.join([tag+';' for tag in _find_img_tag(image_out)])
        print(record['Image Tags'])
        record['Image Description'] = ''
        database.add_record(record)
    database.export_xlsx(database_xlsx, up_freezed_rows=1, left_freezed_cols=2)

if __name__ == '__main__':
    print('Collecting all images...')
    #collect_ffhq_thumbnail_128x128()
    collect_celeba_218x178()


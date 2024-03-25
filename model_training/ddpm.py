import math
import numpy as np
import imageio as iio
import torch
import torch.nn as nn
from torch.utils import data
from typing import Union
from torch.optim.optimizer import Optimizer as Optimizer
from diffusion.model_training.models.diff_unet import Diffusion_UNet
from torch.optim import Adam as AdamOptimizer
from torch.optim.lr_scheduler import LambdaLR
from diffusion.utilities.database import Database
from diffusion.utilities.base_trainer import ModelTrainer
from diffusion.utilities.misc import minibar, Timer, printx
from diffusion.utilities.file_ops import gn, gd, mkdir, join_path, file_exist
from diffusion.utilities.plot import multi_curve_plot
from diffusion.utilities.data_io import load_pkl, save_pkl

'''
Implementation of the original DDPM paper: "Denoising Diffusion Probabilistic Models".
See https://arxiv.org/pdf/2006.11239.pdf for more details.
'''

def save_tensor_to_image(tensor: Union[torch.Tensor, np.ndarray], image_path: str, 
                         intensity_range=[0, 1]):
    '''
    Description
    -----------
    Utility function for saving 2D, 3D or 4D tensor or ndarray into image.
    Input tensor intensity range (min, max) is indicated by `intensity_range`, 
    min will be treated as 0 and max as 255. Outlier values will be clipped.

    Parameters
    -----------
    @param tensor: torch.Tensor | np.ndarray 
        Tensor or ndarray that is about to be saved. 
        Its data format can be:
        * 2D tensor: [h, w]      , grayscale image
        * 3D tensor: [3, h, w]   , RGB image
        * 4D tensor: [b, 3, h, w], multiple RGB images

    @param image_path: str
        Output image path.        
    '''
    assert isinstance(tensor, (torch.Tensor, np.ndarray)), \
        '"tensor" should be of type "torch.Tensor" or "numpy.ndarray", ' \
        'but got "%s".' % (type(tensor).__name__)
    if isinstance(tensor, torch.Tensor):
        ndarr = tensor.detach().cpu().numpy().astype('float32')
    else:
        ndarr = tensor
    ndarr_dim = len(ndarr.shape)
    assert ndarr_dim in [2, 3, 4], 'Only support dumping 2D~4D tensors, got %dD tensor with shape: %s.' % (ndarr_dim, str(ndarr.shape))
    mkdir(gd(image_path))

    ndarr = np.where(ndarr < intensity_range[0], intensity_range[0], ndarr)
    ndarr = np.where(ndarr > intensity_range[1], intensity_range[1], ndarr)
    shift, scale = -intensity_range[0], 1/(intensity_range[1]-intensity_range[0])
    ndarr = (ndarr+shift) * scale # to [0, 1]
    ndarr = (255.0 * ndarr).astype('uint8')

    if ndarr_dim == 2:
        iio.imsave(image_path, ndarr)
    elif ndarr_dim == 3:
        iio.imsave(image_path, np.transpose(ndarr, [1,2,0]))
    else: # ndarr_dim == 4
        # stitch images
        b = ndarr.shape[0]
        n = math.ceil(math.sqrt(b))
        h, w = ndarr.shape[2], ndarr.shape[3]
        stitched_ndarr = np.zeros([3, h * n, w * n], dtype='uint8')
        for row in range(n):
            for col in range(n):
                img_id = row*n+col
                if img_id < b:
                    stitched_ndarr[:, h*row: h*(row+1), w*col: w*(col+1)] = ndarr[img_id, :, :, :]
        iio.imsave(image_path, np.transpose(stitched_ndarr, [1,2,0]))

def roll_dice(prob):
    return torch.rand(()) < prob

class DDPM:
    '''
    Implementation of the DDPM core algorithm.
    '''
    @property
    def max_tval(self):
        ''' max timesteps (T) '''
        return 1000

    @property
    def temb_dim(self):
        ''' length of the time embedding '''
        return self._temb_dim

    @property
    def beta(self):
        ''' using self.beta[t] to get the value of beta_t. '''
        return self._beta_t
    
    @property
    def alpha(self):
        ''' using self.alpha[t] to get the value of alpha_t. '''
        return self._alpha_t

    @property
    def alpha_bar(self):
        ''' using self.alpha_bar[t] to get the value of alpha_bar_t. '''
        return self._alpha_bar_t

    def __init__(self, temb_dim=1024):

        def _exp_schedule(t):
            return 0.02 / (math.e-1) * (math.exp(t)-1)

        t_normed = np.linspace(0.0, 1.0, self.max_tval)

        self._beta_t = np.array( [ _exp_schedule(t) for t in t_normed ] )
        self._alpha_t = 1.0 - self._beta_t
        self._alpha_bar_t = np.cumprod(self._alpha_t)

        # note that alpha[0], alpha_bar[0], and beta[0] is meaningless, 
        # so we insert None below.
        self._alpha_t = [None] + list(self._alpha_t) 
        self._beta_t = [None] + list(self._beta_t)
        self._alpha_bar_t = [None] + list(self._alpha_bar_t)

        self._temb_dim = temb_dim

    def generate_time_embedding(self, tval, return_numpy=False):
        ''' Generate time embedding array from a scalar value. '''
        half_dim = self.temb_dim // 2
        tval = (tval * torch.ones(1)).float()
        temb = math.log(10000) / (half_dim - 1)
        temb = tval * torch.exp(torch.arange(half_dim) * -temb)
        temb = torch.concat([torch.sin(temb), torch.cos(temb)], dim=-1)
        return temb.numpy() if return_numpy else temb
    
    @staticmethod
    def normalize_image(x):
        '''
        Normalize a single image [h, w], a batch of images [b, h, w], 
        or a batch of multi-channel images [b, c, h, w] so that image 
        intensity is in [-1, +1].
        '''
        assert len(x.shape) in [2, 3, 4], \
            'Expected input image to have shape [h, w], [b, h, w], or [b, c, h, w], got %s.' % \
            str([int(dim_size) for dim_size in x.shape])
        assert isinstance(x, (np.ndarray, torch.Tensor)), 'Invalid input image type. Should be "np.ndarray" or "torch.Tensor".'
        where_impl = np.where if isinstance(x, np.ndarray) else torch.where
        min_impl   = np.min   if isinstance(x, np.ndarray) else torch.min
        max_impl   = np.max   if isinstance(x, np.ndarray) else torch.max
        x0 = x.astype('float32') if isinstance(x, np.ndarray) else x.float()
        assert min_impl(x0) > -0.01 and max_impl(x0) < 255.1, \
            'Input image intensity should in range [0, 255], out-of-bound intensity values will be clipped.'
        x0 = where_impl(x0 < 0, 0, x0)
        x0 = where_impl(x0 > 255, 255, x0)
        x0 = x0 / 255.0 # [0, 1]
        x0 = 2 * x0 - 1 # [-1, +1]
        return x0
    
    @staticmethod
    def denormalize_image(x):
        '''
        Denormalize a single image [h, w], a batch of images [b, h, w],
        or a batch of multi-channel images [b, c, h, w] so that image
        intensity range is in [0, 255] (uint8).
        '''
        assert len(x.shape) in [2, 3, 4], \
            'Expected input image to have shape [h, w], [b, h, w], or [b, c, h, w], got %s.' % \
            str([int(dim_size) for dim_size in x.shape])
        assert isinstance(x, (np.ndarray, torch.Tensor)), 'Invalid input image type.'
        where_impl = np.where if isinstance(x, np.ndarray) else torch.where
        x0 = x.astype('float32') if isinstance(x, np.ndarray) else x.float()
        x0 = where_impl(x0 < -1, -1, x0)
        x0 = where_impl(x0 > +1, +1, x0)
        x0 = 255 * ((x0 + 1) / 2)
        x0 = x0.astype('uint8') if isinstance(x, np.ndarray) else x0.byte()
        return x0
    
    def diffusion_process_from_x_0(self, x_0, tval):
        '''
        Forward diffusion: calculate `x_t` from `x_0`.
        '''
        assert isinstance(x_0, (np.ndarray, torch.Tensor)), 'Invalid input image type.'
        n = np.random.randn(*x_0.shape) if isinstance(x_0, np.ndarray) else torch.randn_like(x_0)
        sqrt_impl = np.sqrt if isinstance(x_0, np.ndarray) else (lambda x: torch.sqrt(torch.tensor(x)))
        x_t = sqrt_impl(self._alpha_bar_t[tval]) * x_0 + sqrt_impl(1-self._alpha_bar_t[tval]) * n
        x_t = x_t.astype('float32') if isinstance(x_0, np.ndarray) else x_t.float() 
        n = n.astype('float32') if isinstance(x_0, np.ndarray) else n.float()
        return x_t, n
    
    def reverse_process_from_x_t(self, x_t, tval, n_theta):
        '''
        Backward diffusion: calculate `x_{t-1}` from `x_t` and estimated noise `n_theta`.
        '''
        assert isinstance(x_t, (np.ndarray, torch.Tensor)), 'Invalid input image type.'
        assert isinstance(n_theta, (np.ndarray, torch.Tensor)), 'Invalid input image type.'
        sqrt_impl = np.sqrt if isinstance(x_t, np.ndarray) else (lambda x: torch.sqrt(torch.tensor(x)))
        where_impl = np.where if isinstance(x_t, np.ndarray) else torch.where
        randn_impl = (lambda x: np.random.randn(*x.shape)) if isinstance(x_t, np.ndarray) else (lambda x: torch.randn_like(x))
        # reconstruct (estimate) x_0 from current x_t, then calculate x_{t-1} using estimated x_0 and x_t.
        x_0_pred = 1 / sqrt_impl(self.alpha_bar[tval]) * (x_t - sqrt_impl(1-self.alpha_bar[tval]) * n_theta)
        x_0_pred = where_impl(x_0_pred < -1, -1, x_0_pred)
        x_0_pred = where_impl(x_0_pred > +1, +1, x_0_pred)
        z = randn_impl(x_t)
        if tval > 1:
            mu = (sqrt_impl(self.alpha[tval]) * (1-self.alpha_bar[tval-1])) / (1-self.alpha_bar[tval]) * x_t + \
                self.beta[tval] * sqrt_impl(self.alpha_bar[tval-1]) / (1-self.alpha_bar[tval]) * x_0_pred
            std = sqrt_impl((1-self.alpha_bar[tval-1])/(1-self.alpha_bar[tval]) * self.beta[tval])
        else: 
            # tval==1, then recover from x_1 to x_0, perform noise-free reconstruction.
            mu = x_0_pred
            std = 0
        return mu + std * z
        
    def generate_x_0_from_x_T(self, model, x_T, output_image, tag_ids=None):
        '''
        Generate x_0 from pure Gaussian noise x_T and save to disk as `output_image`.
        We assume x_T ~ N(0, I).
        '''
        assert len(x_T.shape) == 4, 'Input image x_T should have shape [b, 3, h, w].'
        assert isinstance(x_T, (np.ndarray, torch.Tensor)), 'Invalid input image type.'
        assert isinstance(model, nn.Module), 'model should be "nn.Module".'
        if tag_ids is not None:
            assert len(tag_ids.shape) == 2, 'tag_ids should have shape [b, tag_num].'

        mkdir(gd(output_image))
        model.eval()
        if isinstance(x_T, np.ndarray):
            x_T = torch.Tensor(x_T)
        if isinstance(tag_ids, np.ndarray):
            tag_ids = torch.Tensor(tag_ids).long()
        device = next(model.parameters()).device
        x_t = x_T.to(device=device, dtype=torch.float32)
        tag_ids = tag_ids.to(device, dtype=torch.long)
        temb = torch.zeros(x_T.shape[0], self.temb_dim).to(device=device, dtype=torch.float32)
        timer = Timer()
        for tval in reversed(range(1, self.max_tval+1)): # from self.t_max to 1
            temb[:, :] = self.generate_time_embedding(tval).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                n_theta = model(x_t, temb, tag_ids)
            x_t = self.reverse_process_from_x_t(x_t, tval, n_theta)
            minibar('Sampling [%d/%d]' % (self.max_tval-tval+1, self.max_tval), 
                    a=self.max_tval-tval+1, b=self.max_tval, time=timer.elapsed())
        save_tensor_to_image(x_t, output_image, intensity_range=[-1, 1]) # now x_t is x_0
        printx('')

    def demo_diffusion_process(self, image_path, output_folder):
        '''
        Demonstrate the diffusion process using the given `image_path`.
        Generated images will be saved to `output_folder`.
        '''
        x_0 = np.transpose(iio.imread(image_path), [2, 0, 1])
        x_0 = self.normalize_image(x_0)
        t_all = [t for t in range(0, self.max_tval+1, 10)]
        t_all[0] = 1
        mkdir(output_folder)
        for t in t_all:
            x_t, n = self.diffusion_process_from_x_0(x_0, t)
            x_t = np.transpose(self.denormalize_image(x_t), [1, 2, 0])
            iio.imsave(join_path(output_folder, 'x_%d.png' % t), x_t)
        n = np.transpose(self.denormalize_image(n), [1, 2, 0])
        iio.imsave(join_path(output_folder, 'n.png'), n)
        tvals = [t for t in range(0, self.max_tval+1)]
        curve_dict = {
            r'$\beta_t$': {'x': tvals, 'y': self.beta, 'label': True, 'color': [0.9,0.1,0.1]},
            r'$\alpha_t$': {'x': tvals, 'y': self.alpha, 'label': True, 'color': [0.3,0.7,0.3]},
            r'$\bar\alpha_t$': {'x': tvals, 'y': self.alpha_bar, 'label': True, 'color': [0.1,0.1,0.9]},
        }
        multi_curve_plot(curve_dict=curve_dict, save_file=join_path(output_folder,'constants.pdf'), 
                         fig_size=(5,5), dpi=150, title='Constants used in the diffusion process', 
                         xlabel=r'$t$', ylabel='Value')

class DDPM_Loader(torch.utils.data.Dataset):
    ''' Implementation of image loader used in DDPM training process. '''
    # this defines the database structure
    db_keys = ['Image Source', 'Image Name', 'Image Path', 'Image Tags', 'Image Description']
    image_tag_to_id = {
        'N/A': 0,
        # CelebA dataset
        '5_o_Clock_Shadow': 1, 'Arched_Eyebrows': 2, 'Attractive': 3, 'Bags_Under_Eyes': 4, 
        'Bald Bangs': 5, 'Big_Lips': 6, 'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9,
        'Blurry': 10, 'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13, 'Double_Chin': 14, 
        'Eyeglasses': 15, 'Goatee': 16, 'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
        'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22, 'Narrow_Eyes': 23, 'No_Beard': 24,
        'Oval_Face': 25, 'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28, 
        'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31, 'Straight_Hair': 32, 'Wavy_Hair': 33,
        'Wearing_Earrings': 34, 'Wearing_Hat': 35, 'Wearing_Lipstick': 36, 'Wearing_Necklace': 37,
        'Wearing_Necktie': 38, 'Young': 39, 
        # Extended tags
        'Female': 40,
    }
    max_tags_to_keep = 16
    num_total_tags = 64

    @staticmethod
    def _generate_image_ID_from_record(record):
        return record['Image Source'] + '@' + record['Image Name']

    @staticmethod
    def _get_tag_ids_from_tag_string(tag_string):
        tags = [tag for tag in tag_string.strip().split(';') if tag != '']
        if 'Male' not in tags:
            tags.append('Female')
        # randomly drop tags for augmentation (20% drop rate for each tag)
        # but if the tag is 'Male' or 'Female', we keep it without dropping.
        tags = [tag for tag in tags if roll_dice(0.8) or tag in ['Male', 'Female']]
        assert 'Male' in tags or 'Female' in tags
        tag_ids = [ DDPM_Loader.image_tag_to_id[tag] for tag in tags if tag in DDPM_Loader.image_tag_to_id]
        tag_ids = sorted(tag_ids, reverse=False)
        if len(tag_ids) < DDPM_Loader.max_tags_to_keep:
            tag_ids += [0] * (DDPM_Loader.max_tags_to_keep - len(tag_ids))
        tag_ids = torch.Tensor(tag_ids).long()
        # random drop tags and keep tag count to max_tags_to_keep
        def _stable_keep_topk(arr, k):
            # random select k samples from arr, but keep 
            # the relevant orders between elements (thus 'stable').
            assert len(arr) >= k and torch.is_tensor(arr)
            indices = torch.argsort(torch.randn(len(arr)), descending=True)[:k]
            indices = torch.sort(indices)[0]
            return arr[indices]
        tag_ids = _stable_keep_topk(tag_ids, DDPM_Loader.max_tags_to_keep)
        assert len(tag_ids) == DDPM_Loader.max_tags_to_keep
        return tag_ids

    def _load_image_to_mem(self, image_record):
        '''  load image and apply preprocessing strategies '''
        image_path = image_record['Image Path']
        x_0 = iio.imread(image_path)
        x_0 = self.ddpm.normalize_image(x_0)
        x_0 = np.transpose(x_0, [2, 0, 1]) # [h, w, c] -> [c, h, w]
        return x_0
    
    def __init__(self, xlsx_file, images_per_epoch=None, use_cache=False, 
                 apply_DA=False, ddpm_impl=DDPM()):
        self.ddpm = ddpm_impl
        self.apply_DA = apply_DA # data augmentation
        self.images_per_epoch = images_per_epoch
        self.database = Database(db_keys=DDPM_Loader.db_keys, xlsx_file=xlsx_file)
        self.load_to_memory = use_cache
        self.loaded_data = {}
        timer = Timer()
        cached_data_pack = gn(xlsx_file, no_extension=True) + '.cache'
        if use_cache and file_exist(cached_data_pack):
            printx('Loading cached data "%s"...' % cached_data_pack)
            self.loaded_data = load_pkl(cached_data_pack)
            printx('')
            print('Loaded cached data "%s".' % cached_data_pack)
        else:
            for record_id in range(self.database.num_records()):
                record = self.database.get_record(record_id)
                image_data = self._load_image_to_mem(record) if use_cache else None
                image_ID = self._generate_image_ID_from_record(record)
                self.loaded_data[image_ID] = {'image_record': record, 'image_data': image_data}
                minibar(msg='Loading images (%s) [%d/%d]' % \
                        ("disk->mem" if use_cache else "disk",
                         record_id+1,self.database.num_records()), 
                        a=record_id+1, b=self.database.num_records(), time=timer.elapsed())
            printx('') # clear output
        if use_cache and not file_exist(cached_data_pack):
            # pack all data into a single pkl file
            save_pkl(self.loaded_data, cached_data_pack)
            print('Loaded data cached to "%s".' % cached_data_pack)
    
    def __getitem__(self, index):
        def rand_int(start, end):
            return int(torch.randint(start, end, (1,)))
        if self.images_per_epoch:
            # randomly sample an image from the whole database
            sample_id = rand_int(0, self.database.num_records())
        else:
            # sample image using the given index 
            # as the whole database is used for training
            sample_id = index
        # retrieve image data
        image_record = self.database.get_record(sample_id)
        image_ID = self._generate_image_ID_from_record(image_record)
        image_tag_string = image_record['Image Tags']
        image_tag_ids = DDPM_Loader._get_tag_ids_from_tag_string(image_tag_string)
        image_tag_ids = np.array(image_tag_ids, dtype=np.int32)
        if self.load_to_memory: # data already loaded into memory
            x_0 = self.loaded_data[image_ID]['image_data']
        else:
            x_0 = self._load_image_to_mem(image_record)
        # apply data augmentation if required
        if self.apply_DA:
            if roll_dice(0.5):
                # flip image horizontally with 50% percentage 
                x_0 = x_0[:, :, ::-1]
        # do forward process and generate time embedding
        tval = rand_int(1, self.ddpm.max_tval+1)
        x_t, noise = self.ddpm.diffusion_process_from_x_0(x_0, tval)
        tval = self.ddpm.generate_time_embedding(tval)
        return image_record, x_t, tval, image_tag_ids, noise
    
    def __len__(self):
        if self.images_per_epoch is None:
            return self.database.num_records()  
        else:
            return self.images_per_epoch

class DDPM_Trainer(ModelTrainer):
    
    def __init__(self, train_configs, model_configs,
                 train_xlsx=None, val_xlsx=None, test_xlsx=None):

        ddpm = DDPM(temb_dim=model_configs['temb_dims'][0])
        model_configs['n_tags'] = DDPM_Loader.num_total_tags
        model = Diffusion_UNet(**model_configs).cuda(train_configs['gpu_index'])
        optim = AdamOptimizer(model.parameters(), lr=train_configs['initial_lr'], betas=(0.9, 0.999))
        lr_scheduler = LambdaLR(optim, lambda epoch: 0.5**(epoch//50)) # lr cut to half per 50 epochs
        print('# trainable params:', model.nparams(as_string=True))

        train_loader = data.DataLoader(
            DDPM_Loader(train_xlsx, use_cache=train_configs['use_cache'], 
                        ddpm_impl=ddpm, apply_DA=train_configs['apply_train_DA']), 
            batch_size=train_configs['train_batch_size'], shuffle=True, 
            num_workers=train_configs['dataloader_workers']) if train_xlsx else None
        val_loader = data.DataLoader(
            DDPM_Loader(val_xlsx, use_cache=train_configs['use_cache'], 
                        ddpm_impl=ddpm, apply_DA=train_configs['apply_val_DA']), 
            batch_size=train_configs['val_batch_size'], shuffle=False, 
            num_workers=train_configs['dataloader_workers']) if val_xlsx else None
        test_loader = data.DataLoader(
            DDPM_Loader(test_xlsx, use_cache=train_configs['use_cache'], 
                        ddpm_impl=ddpm, apply_DA=train_configs['apply_test_DA']), 
            batch_size=train_configs['test_batch_size'], shuffle=False, 
            num_workers=train_configs['dataloader_workers']) if test_xlsx else None

        super().__init__(train_configs['model_folder'], model, optim, 
            lr_scheduler, train_loader, val_loader, test_loader,
            train_configs['save_every_n_epochs'])

        self.ddpm = ddpm
        self.train_configs = train_configs
        self.model_configs = model_configs

        save_pkl(train_configs, self.output_folder+'/train_configs.pkl')
        save_pkl(model_configs, self.output_folder+'/model_configs.pkl')
    
    def _to_assigned_device(self, *inputs, device=None):
        assert device is not None, '`device` should not be None.'
        device_inputs = []
        for i, _ in enumerate(inputs): 
            device_inputs.append(inputs[i].to(device))
        return device_inputs

    @staticmethod
    def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.shape == y_true.shape
        numel = torch.numel(y_pred)
        return torch.sum(torch.pow(y_pred-y_true, 2)) / numel
    
    def _forward_pass(self, *inputs):
        def _get_device():
            return next(self.model.parameters()).device
        
        x_t, t, tag_ids, n = self._to_assigned_device(*inputs, device=_get_device())
        n_theta = self.model(x_t, t, tag_ids=tag_ids if roll_dice(0.1) else None)
        loss = self.mse_loss(n_theta, n)
        return loss
    
    def _on_epoch(self, epoch, msg, data_loader, phase):
        timer = Timer()
        batch_losses = []
        for batch_idx, (batch_data) in enumerate(data_loader):
            if phase in ['train']:
                self.model.train()
                self.optim.zero_grad()
                loss = self._forward_pass(*batch_data[1:])
                loss.backward()
                self.optim.step()
            elif phase in ['val', 'test']:
                self.model.eval()
                with torch.no_grad():
                    loss = self._forward_pass(*batch_data[1:])
            batch_losses.append(loss.item())
            minibar(msg=msg, a=batch_idx+1, b=len(data_loader), 
                    time=timer.elapsed(), last='%.4f' % float(loss.item()))
        return np.log10(np.mean(batch_losses))
    
    def _generate_tag_ids_for_CelebA(self):
        train_batch_size = self.train_configs['train_batch_size']
        tag_ids = np.zeros([train_batch_size, DDPM_Loader.max_tags_to_keep], dtype=np.int32)
        tag_ids[:10, 0] = DDPM_Loader.image_tag_to_id['Female']
        tag_ids[:10, 1] = DDPM_Loader.image_tag_to_id['Attractive']
        tag_ids[:10, 2] = DDPM_Loader.image_tag_to_id['Smiling']
        tag_ids[:10, 3] = DDPM_Loader.image_tag_to_id['Black_Hair']
        tag_ids[:10, 4] = DDPM_Loader.image_tag_to_id['Heavy_Makeup']
        tag_ids[10:20, 0] = DDPM_Loader.image_tag_to_id['Male']
        tag_ids[10:20, 1] = DDPM_Loader.image_tag_to_id['Smiling']
        tag_ids[10:20, 2] = DDPM_Loader.image_tag_to_id['Mustache']
        tag_ids[10:20, 3] = DDPM_Loader.image_tag_to_id['Big_Nose']
        tag_ids[10:20, 4] = DDPM_Loader.image_tag_to_id['Wearing_Hat']
        tag_ids = np.where(tag_ids == 0, 1e8, tag_ids)
        tag_ids = np.sort(tag_ids, axis=-1)
        tag_ids = np.where(tag_ids == 1e8, 0, tag_ids)
        return tag_ids

    def _on_epoch_end(self, epoch: int):
        train_batch_size = self.train_configs['train_batch_size']
        self.ddpm.generate_x_0_from_x_T(
            self.model, np.random.randn(train_batch_size, 3, 128, 128),
            join_path(self.output_folder, 'preview', 'epoch_%04d.png' % epoch),
            tag_ids=self._generate_tag_ids_for_CelebA())
    
    def load_checkpoint(self, ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        self.model.load_state_dict( checkpoint['model_state_dict'], strict=True)
        self.optim.load_state_dict( checkpoint['optim_state_dict'])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict( checkpoint['lr_sched_state_dict'])
        self.model = self.model.cuda(self.train_configs['gpu_index'])
        self.train_states = load_pkl(ckpt_file + '.train_states')

    def save_checkpoint(self, ckpt_file):
        mkdir(gd(ckpt_file))
        save_pkl(self.train_states, ckpt_file + '.train_states')
        checkpoint = {
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optim_state_dict': self.optim.state_dict(),
            'lr_sched_state_dict': self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, ckpt_file)

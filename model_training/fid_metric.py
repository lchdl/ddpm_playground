import os
import torch
import numpy as np
import imageio as iio
from scipy.linalg import sqrtm
from scipy.ndimage import zoom
from diffusion.model_training.models.inception import inception_v3, Inception3
from diffusion.utilities.file_ops import file_exist

def resize_image_for_inception_v3(x: np.ndarray, order=3, return_type='float32'):
    '''
    Resize image with shape [3, h, w] into shape [3, 299, 299] 
    as required by inception_v3 model architecture.
    '''
    assert len(x.shape) == 3 and x.shape[0] == 3, \
        'Only accepts image with shape [3, h, w], got %s.' % \
        str([int(dim_sz) for dim_sz in x.shape])
    ch_r, ch_g, ch_b, h, w = x[0], x[1], x[2], x.shape[1], x.shape[2]
    scale_h, scale_w = 299 / h, 299 / w
    ch_r = zoom(ch_r, (scale_h, scale_w), order=order)
    ch_g = zoom(ch_g, (scale_h, scale_w), order=order)
    ch_b = zoom(ch_b, (scale_h, scale_w), order=order)
    x0 = np.zeros([3, 299, 299])
    def min(a, b):
        return a if a < b else b
    new_h, new_w = min(299, ch_r.shape[0]), min(299, ch_r.shape[1])
    x0[0, :new_h, :new_w] = ch_r[:new_h, :new_w]
    x0[1, :new_h, :new_w] = ch_g[:new_h, :new_w]
    x0[2, :new_h, :new_w] = ch_b[:new_h, :new_w]
    x0 = x0.astype(return_type)
    return x0

class FID_Metric:
    def __init__(self, model_cache='', gpu_index=0):
        self.gpu_index = gpu_index
        # load inception_v3
        self.model: Inception3 = None
        if not file_exist(model_cache):
            self.model = inception_v3(pretrained=True, transform_input=True)
            if model_cache != '':
                torch.save(self.model.state_dict(), model_cache)
        else:
            self.model = inception_v3(pretrained=False, transform_input=True)
            self.model.load_state_dict(torch.load(model_cache), strict=False)
        self.model.cuda(gpu_index)
    def _fid_inception_v3(self, feats1: np.ndarray, feats2: np.ndarray):
        '''
        Calculate FID score between two sets of activations feats1 and feats1.
        feats1: [b, 2048], feats2: [b, 2048], where b is the batch size.
        *: feats1 and feats2 should be the outputs from the inception_v3 model.
        '''
        print([feats1[:8,:8]],'\n')
        print([feats2[:8,:8]])
        # calculate mean and covariance statistics
        mu1, sigma1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
        mu2, sigma2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(np.dot(sigma1,sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    def _setup_env(self):
        # See https://github.com/scipy/scipy/issues/14594.
        if 'MKL_NUM_THREADS' not in os.environ:
            os.environ['MKL_NUM_THREADS'] = "1"
            self._should_delete_MKL_NUM_THREADS = True
        else:
            self._should_delete_MKL_NUM_THREADS = False
            self._prev_MKL_NUM_THREADS = os.environ['MKL_NUM_THREADS']
            os.environ['MKL_NUM_THREADS'] = "1"        
    def _cleanup_env(self):
        if self._should_delete_MKL_NUM_THREADS:
            os.environ.pop('MKL_NUM_THREADS')
        else:
            os.environ['MKL_NUM_THREADS'] = self._prev_MKL_NUM_THREADS
    def fid(self, images1, images2, batch_size=32):
        '''
        Calculate FID score between two sets of images (images1, images2).
        Each set of images is represented by a list of image paths.
        '''
        assert len(images1) == len(images2), \
            'The number of images from the two sets must be identical, '\
            'got %d and %d.' % (len(images1), len(images2))
        assert batch_size > 1, 'Batch size must > 1.'
        n = batch_size * (len(images1) // batch_size)
        if n <= 0:
            raise RuntimeError('The number of images (%d) is smaller than batch size (%d). '
                               'Try decrease batch size or increase your image count.' % \
                                (len(images1), batch_size))
        feats1 = np.zeros([n, 2048], dtype='float32')
        feats2 = np.zeros([n, 2048], dtype='float32')
        def _get_image_batch_feature(image_batch):
            x = np.zeros([batch_size, 3, 299, 299], dtype='float32')
            for i, image_path in enumerate(image_batch):
                x0 = iio.imread(image_path) # [h, w, 3]
                x0 = np.transpose(x0, [2, 0, 1])
                x0 = resize_image_for_inception_v3(x0, return_type='float32')
                x[i] = x0
            x = torch.Tensor(x).cuda(self.gpu_index)
            self.model.eval()
            with torch.no_grad():
                self.model(x) # simply call forward and cache feature
            feats = self.model.feature_cache(to_numpy=True) # [b, 1024]
            return feats
        for i_start in range(0, n, batch_size):
            i_end = i_start + batch_size
            feats1[i_start:i_end, :] = _get_image_batch_feature(images1[i_start:i_end])
            feats2[i_start:i_end, :] = _get_image_batch_feature(images2[i_start:i_end])
        self._setup_env()
        fid_metric = self._fid_inception_v3(feats1, feats2)
        self._cleanup_env()
        return fid_metric

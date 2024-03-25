import torch
import numpy as np
from typing import Union
from torch.utils import data
from torch.optim.optimizer import Optimizer as Optimizer
from diffusion.utilities.file_ops import file_exist, mkdir, join_path, gd, gn
from diffusion.utilities.data_io import load_pkl, save_pkl
from diffusion.utilities.misc import Timer, printx, format_sec
from diffusion.utilities.plot import multi_curve_plot

class ModelTrainer:
    '''
    Description
    -----------
    Simple utility class for training a model under PyTorch deep learning framework with
    simple error checking mechanics. This also helps users to avoid some common errors 
    when designing a network trainer from scratch.
    This template class also implements the following features:
        * Automatic model loading/saving
        * Automatic adjust learning rate after every epoch (if lr_scheduler is given)
        * Saving the current best model using validation set (if given)
        * Plot network training process
        * Event driven design
            - override _on_epoch(...) in derived class to define how the model is trained
              in train/validation/test phase.
            - override _on_load_model(...) in derived class when model is being loaded
            - override _on_save_model(...) in derived class when model is being saved
            - ...
            
    Note
    -----------
    This class is a template class. Inherit this class and at least override & implement 
    ModelTrainer_PyTorch::_on_epoch(...) in derived class to make it fully usable. See 
    docstring of _on_epoch(...) for more info.

    '''
    def __init__(self, output_folder='',
                 model=None, optim=None, lr_scheduler=None,
                 train_loader=None, val_loader=None, test_loader=None,
                 save_every_n_epochs=None,
    ):
        '''
        Parameters
        -----------
        @param `output_folder`: 
            Model output directory. All outputs from the model trainer will be saved to here.
        @param `model`:
            The model waiting to be trained. The model object should be an instance of 
            `torch.nn.Module` and needs to be moved to the proper device(s) (using `.to()`,
            `.cuda()`, etc.) before initiating the training process.
        @param `optim`:
            Model optimizer. `optim` should already be linked to the `model` before training.
        @param `lr_scheduler`: 
            [optional] Learning rate scheduler. If ignored, constant learning rate will be used.
        @param `train_loader`:
            [optional] Dataloader object used in training phase. It should be an instance of 
            `torch.utils.data.Dataloader`
        @param `val_loader`:
            [optional] Dataloader object used in validation phase. It should be an instance of 
            `torch.utils.data.Dataloader`
        @param `test_loader`:
            [optional] Dataloader object used in test phase. It should be an instance of 
            `torch.utils.data.Dataloader`
        @param `save_every_n_epochs`: 
            [optional] Saving the model at scheduled intervals during training. Set it to `None` 
            to disable.
        '''
        assert model is not None, '`model` should not be None.'
        assert optim is not None, '`optim` should not be None.'

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        self.output_folder = mkdir(output_folder)
        self.train_states = {
            'cur_epoch':1, 'train_loss':[], 'val_loss':[], 'test_loss':[], 'best_loss':None
        }

        self.model = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.save_every_n_epochs = save_every_n_epochs

    def load_checkpoint(self, ckpt_file):
        raise NotImplementedError('Unimplemented virtual function')
    
    def save_checkpoint(self, ckpt_file):
        raise NotImplementedError('Unimplemented virtual function')

    def plot_training_progress(self, output_image):
        train_loss = [(float(s) if s != None else np.nan) for s in self.train_states['train_loss']]
        val_loss   = [(float(s) if s != None else np.nan) for s in self.train_states['val_loss']]
        test_loss  = [(float(s) if s != None else np.nan) for s in self.train_states['test_loss']]
        epochs = [i for i in range(1, len(train_loss)+1)]
        curves = {
            'training'  : { 'x': epochs, 'y': train_loss, 'color': [0.0, 0.5, 0.0], 'label': True },
            'validation': { 'x': epochs, 'y': val_loss,   'color': [0.0, 0.0, 1.0], 'label': True },
            'test'      : { 'x': epochs, 'y': test_loss,  'color': [1.0, 0.0, 0.0], 'label': True },
        }
        mkdir(gd(output_image))
        multi_curve_plot(curves, output_image, dpi=150, 
                            title='Training Progress', 
                            xlabel='Epoch', ylabel='Loss')

    def get_current_lr(self):
        return self.optim.param_groups[0]["lr"]

    def train(self, num_epochs=1000, print_mode='all'):
        '''
        Description
        ------------
        Train the model using the given data loader, optimizer, lr scheduler (optional).

        Parameters
        ------------
        @param num_epochs: int
            Number of training epochs.
        @param print_mode: str, ("all" | "both")
            If set to "all", then training results of all epochs will be printed.
            If set to "best", then only the current best epoch will be printed.
        
        Note
        ------------
        * This function will call _on_epoch(...), user need to implement _on_epoch(...)
          before calling this.
        '''
        assert print_mode in ['all', 'best'], 'Invalid `print_mode` setting.'

        ckpt_latest = join_path(self.output_folder, 'latest.ckpt')
        ckpt_best = join_path(self.output_folder, 'best.ckpt')
        if file_exist(ckpt_latest):
            print('* Loading latest checkpoint ...')
            self.load_checkpoint(ckpt_latest)
        elif file_exist(ckpt_best):
            print('* Latest checkpoint not found, loading current best checkpoint ...')
            self.load_checkpoint(ckpt_best)
        else:
            print('* Training model from scratch.')

        start_epoch, end_epoch = self.train_states['cur_epoch'], num_epochs+1
        timer = Timer()

        self._on_train_start()

        def _table_style():
            return [
                '====================================================',
                ' EPOCH    TRAIN     VALIDATION     TEST     ELAPSED ',
                '----------------------------------------------------',
                ' *        *           *           *         *       ',
            ]
        
        print(_table_style()[0])
        print(_table_style()[1])
        print(_table_style()[2])

        printx('Trainer is launching, please wait...')

        for epoch in range(start_epoch, end_epoch):
            timer.tick()

            if self.lr_scheduler is not None:
                lr_before_epoch_start = self.get_current_lr()
            self._on_epoch_start(epoch)

            train_fetch = self._on_epoch(epoch, 
                msg='Epoch %d/%d 1/3' % (epoch, num_epochs), 
                data_loader=self.train_loader, 
                phase='train') if self.train_loader else None
            val_fetch = self._on_epoch(epoch, 
                msg='Epoch %d/%d 2/3' % (epoch, num_epochs), 
                data_loader=self.val_loader,
                phase='val') if self.val_loader else None
            test_fetch = self._on_epoch(epoch, 
                msg='Epoch %d/%d 3/3' % (epoch, num_epochs), 
                data_loader=self.test_loader,  
                phase='test') if self.test_loader else None

            if self.lr_scheduler is not None:
                lr_after_epoch_end = self.get_current_lr()
                if lr_before_epoch_start != lr_after_epoch_end:
                    # This means user accidentally changes the learning rate during model training. 
                    # Users do not need to care about learning rate scheduling as it can be handled 
                    # properly by the model trainer.
                    printx('')
                    print('* Warning: lr is changed by user! Normally you don\'t need to care about '
                          'lr scheduling as it will be handled properly by the model trainer.')
                self.lr_scheduler.step()
                lr_after_updated = self.get_current_lr()

            def _check_fetch(any_fetch, phase):
                if any_fetch is None:
                    return None
                assert isinstance(any_fetch, (float, tuple, list)), \
                    'The return value of "self._epoch(...)" in %s phase should be of type '\
                    '"float", "tuple", or "list". Got an instance of type "%s" instead.' % \
                    (phase, type(any_fetch).__name__)
                if isinstance(any_fetch, float):
                    any_fetch = [any_fetch]
                assert isinstance(any_fetch[0], float), \
                    'The FIRST return value of "self._epoch(...)" in %s phase should be a '\
                    'FLOAT scalar, which will be used in model selection if validation set'\
                    ' is available. Got an instance of type "%s" instead.' % \
                    (phase, type(any_fetch).__name__)
                return any_fetch

            train_fetch = _check_fetch(train_fetch, 'train')
            val_fetch   = _check_fetch(val_fetch,   'val')
            test_fetch  = _check_fetch(test_fetch,  'test')

            train_loss = train_fetch[0] if train_fetch is not None else None
            val_loss   = val_fetch[0]   if val_fetch   is not None else None
            test_loss  = test_fetch[0]  if test_fetch  is not None else None

            if val_loss is None:
                is_best = True
            else:
                is_best = self.train_states['best_loss'] is None       \
                          or                                           \
                          val_loss < self.train_states['best_loss']
            if is_best:
                self.train_states['best_loss'] = val_loss
            
            def _print_one_line_epoch_summary():
                def _left_replace_placeholder_once(str_with_placeholder, content, cell_size):
                    if len(content)<cell_size: content += ' '*(cell_size-len(content))
                    elif len(content)>cell_size: content = content[:cell_size-3]+'...'
                    return str_with_placeholder.replace('*'+' '*(cell_size-1), content, 1)

                oneline_summary = _table_style()[3]
                oneline_summary = _left_replace_placeholder_once(oneline_summary, 
                                    '%4d%s' % (epoch, '*' if is_best else ' '), 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, 
                                    ('%1.4f' % train_loss) if train_loss is not None else '  --  ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, 
                                    ('%1.4f' % val_loss)   if val_loss   is not None else '  --  ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, 
                                    ('%1.4f' % test_loss)  if test_loss  is not None else '  --  ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, 
                                    '%s' % format_sec( int(timer.tick()) ), 8)

                printx('')
                print(oneline_summary)
            
            # if user returns additional info, print it
            def _print_additional_info_if_required(any_fetch, note):
                if any_fetch is None or len(any_fetch) < 2: return
                formatted_info = '  * %s: ' % note
                for item in any_fetch[1:]:
                    formatted_info += str(item)  + ' '
                print(formatted_info)

            if print_mode == 'all' or (print_mode == 'best' and is_best):
                # if print_mode=='best' then only the current best will be printed. 
                _print_one_line_epoch_summary()
                _print_additional_info_if_required(train_fetch, 'train')
                _print_additional_info_if_required(val_fetch,   '  val')
                _print_additional_info_if_required(test_fetch,  ' test')

            if self.lr_scheduler is not None:
                if lr_after_epoch_end != lr_after_updated:
                    printx('')
                    print('* Learning rate updated to %f since epoch %d.' % (lr_after_updated, epoch))

            self.train_states['cur_epoch'] += 1
            self.train_states['train_loss'].append(train_loss)
            self.train_states['val_loss'].append(val_loss)
            self.train_states['test_loss'].append(test_loss)
            
            printx('Saving model, please do not kill the program...')
            self.save_checkpoint(ckpt_latest)
            if is_best:
                printx('Saving best model, please do not kill the program...')
                self.save_checkpoint(ckpt_best)
                self._on_best_epoch(epoch, ckpt_best)
            if self.save_every_n_epochs is not None and self.save_every_n_epochs > 0:
                if self.train_states['cur_epoch'] % self.save_every_n_epochs == 0:
                    printx('Scheduled model saving for every %d epoch(s)...' % self.save_every_n_epochs)
                    scheduled_ckpt_save = join_path(self.output_folder, 'epoch_%d.ckpt' % \
                                                    self.train_states['cur_epoch'])
                    self.save_checkpoint(scheduled_ckpt_save)
            printx('Model(s) saved. Visualizing training progress...')
            self.plot_training_progress(join_path(self.output_folder, 'progress.png'))
            self._on_epoch_end(epoch)
            printx('Preparing for next epoch...')

        printx('')
        print(_table_style()[0])

        self._on_train_end()
    
    def _on_train_start(self):
        '''
        Add additional operations when the training process is about to start.
        '''
        pass

    def _on_train_end(self):
        '''
        Add additional operations when the training process is ended.
        '''
        pass

    def _on_epoch_start(self, epoch: int):
        '''
        This will be called when current epoch is about to start training.
        '''
        pass

    def _on_epoch_end(self, epoch: int):
        '''
        This will be called when current epoch training is finished.
        '''
        pass

    def _on_epoch(self, 
            epoch:       int,
            msg:         str                          = '', 
            data_loader: Union[data.DataLoader, None] = None,
            phase:       str                          = '',
        ) -> Union[ float, tuple, list]:
        '''
        Description
        -----------
        `phase`: can be one of "train", "val", or "test" indicating if model 
        is currently under "training", "validation" or "test" mode.
        NOTE: self._on_epoch() can return multiple items, but only the first 
            item will be used for model selection (it is treated as the most 
            important metric and the rest are just additional info, also be 
            aware that the first item should be a scalar value (float).
        '''
        raise NotImplementedError('Unimplemented virtual function '
            '"ModelTrainer_PyTorch::_on_epoch(...)" called. Please '
            'implement it in child class.')

    def _on_best_epoch(self, best_epoch: int, best_model_path: str):
        '''
        This will be called when a current best model is produced and about to be saved.
        '''
        pass

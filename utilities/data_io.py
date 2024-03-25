import os, gzip, tarfile, pickle, json, xlsxwriter, shutil, openpyxl
from typing import Union
from xlsxwriter.format import Format
from diffusion.utilities.file_ops import abs_path, dir_exist, file_exist, join_path, gd, gn

# utility function used to compress a file into "*.gz"
# (not suitable for compressing folders)
def gz_compress(file_path, out_path=None, compress_level:int=9, verbose=False, overwrite = True):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_path), 'file "%s" not exist or it is a directory. '\
        'gzip can be only used to compress files, if you want to compress folder structure, '\
        'please use targz_compress(...) instead.' % file_path
    f = open(file_path,"rb")
    data = f.read()
    bindata = bytearray(data)
    gz_path = join_path( gd(file_path) , gn(file_path) + '.gz' ) if out_path is None else out_path
    if file_exist(gz_path) and not overwrite:
        if verbose:
            print('skip %s' % (file_path))
    else:
        if verbose:
            print('%s >>> %s' % (file_path, gz_path))
        with gzip.GzipFile(filename=gz_path, mode='wb', compresslevel=compress_level) as f:
            f.write(bindata)
    return gz_path

def gz_uncompress(gz_path, out_path=None, verbose=False):
    out_path0 = ''
    if out_path is not None:
        out_path0 = out_path
    else:
        if gz_path[-3:] == '.gz':
            out_path0 = gz_path[:-3]
        else:
            raise RuntimeError(
                'Incorrect gz file name. Input file name must '
                'end with "*.gz" if out_path is not set.')
    out_path0 = abs_path(out_path0)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path0, 'wb') as f_out:
            if verbose: print('%s >>> %s' % (gz_path, out_path))
            shutil.copyfileobj(f_in, f_out)

# compress a file or a folder structure into *.tar.gz format.
def targz_compress(file_or_dir_path, out_file=None, compress_level:int=9, verbose=False):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_or_dir_path) or dir_exist(file_or_dir_path), \
        'file or directory not exist: "%s".' % file_or_dir_path
    targz_path = join_path(gd(file_or_dir_path) , gn(file_or_dir_path) + '.tar.gz') if out_file is None else out_file
    if file_exist(file_or_dir_path):
        # target path is a file
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            tar.add(file_or_dir_path, arcname=gn(file_or_dir_path) )
            if verbose:
                print('>>> %s' % file_or_dir_path)
    elif dir_exist(file_or_dir_path):
        # target path is a folder
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            for name in os.listdir( file_or_dir_path ):
                tar.add( join_path(file_or_dir_path, name) , recursive=True, arcname=name)
                if verbose:
                    print('>>> %s' % name)
    else:
        raise RuntimeError('only file or folder can be compressed.')

def targz_uncompress(targz_file, out_path):
    '''
    out_path: str
        path to output folder.
    '''
    targz = tarfile.open(targz_file)
    targz.extractall(out_path)
    targz.close()

def save_pkl(obj, pkl_path):
    with open(pkl_path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(pkl_path):
    content = None
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
    return content

def save_json(obj, json_path, indent=4):
    with open(json_path,'w') as f:
        json.dump(obj, f,indent=indent)

def load_json(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
    return obj

class SimpleExcelWriter:
    def __init__(self, file_path, worksheet_names='default'):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx".')
        self.file_path = file_path

        if isinstance(worksheet_names, str):
            self.worksheet_names = [worksheet_names]
        elif isinstance(worksheet_names, list):
            self.worksheet_names = worksheet_names
        else:
            raise RuntimeError('Only str or list type are accepted. Got type "%s".' % type(worksheet_names).__name__)

        # create workbook and worksheet(s)
        self.workbook = xlsxwriter.Workbook(self.file_path)
        self.worksheets = {}
        for worksheet_name in self.worksheet_names:
            self.worksheets[worksheet_name] = self.workbook.add_worksheet(worksheet_name)

    def _is_closed(self):
        return self.workbook is None
    
    def _check_closed(self):
        if self._is_closed():
            raise RuntimeError('Excel file is already closed and saved, which cannot be written anymore!')

    def _close(self):
        self.workbook.close()
        self.workbook = None

    def new_format(self,bold=False,italic=False,underline=False,font_color='#000000',bg_color='#FFFFFF'):

        self._check_closed()
        cell_format = self.workbook.add_format()
        cell_format.set_bold(bold)
        cell_format.set_italic(italic)
        cell_format.set_underline(underline)
        cell_format.set_font_color(font_color)
        cell_format.set_bg_color(bg_color)

        return cell_format

    def set_column_width(self, pos, width, worksheet_name='default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(pos, int):
            self.worksheets[worksheet_name].set_column(pos,pos, width)
        elif isinstance(pos,str):
            self.worksheets[worksheet_name].set_column(pos, width)
        elif isinstance(pos,tuple) or isinstance(pos,list):
            assert len(pos)==2, 'Invalid position setting.'
            start,end=pos
            self.worksheets[worksheet_name].set_column(start,end,width)

    def write(self, cell_name_or_pos, content, worksheet_name='default', format=None):
        self._check_closed()        
        if format is not None:
            assert isinstance(format, Format), 'Invalid cell format.'
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(cell_name_or_pos, tuple) or isinstance(cell_name_or_pos, list):
            assert len(cell_name_or_pos) == 2, 'Invalid cell position.'
            row,col = cell_name_or_pos
            if format:
                self.worksheets[worksheet_name].write(row,col,content,format)
            else:
                self.worksheets[worksheet_name].write(row,col,content)
        elif isinstance(cell_name_or_pos, str):
            if format:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content,format)
            else:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content)
        else:
            raise RuntimeError('Invalid cell name or position. Accepted types are: tuple, list or str but got "%s".' % \
                type(cell_name_or_pos).__name__)

    def set_freeze(self, first_n_rows = 0, first_n_cols = 0,  worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].freeze_panes(first_n_rows, first_n_cols)

    def set_zoom(self, zoom_factor = 100, worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].set_zoom(zoom_factor)
    
    def set_filter(self, start_pos, end_pos, worksheet_name = 'default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        self.worksheets[worksheet_name].autofilter(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

    def save_and_close(self):
        self._close()

class SimpleExcelReader:
    def __init__(self, file_path):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx". ".xls" format is not supported.')
        self.file_path = file_path

        self.xlsx = openpyxl.load_workbook(file_path)
    
    def max_row(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_row
    
    def max_column(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_column
    
    def read(self, pos: Union[list, tuple], worksheet_name='default'):
        if self.xlsx is None:
            raise RuntimeError('file is already closed.')
        assert len(pos) == 2, 'invalid cell position'
        pos0 = pos[0]+1, pos[1]+1 # cell index starts with 1
        return self.xlsx[worksheet_name].cell(pos0[0], pos0[1]).value
    
    def close(self):
        self.xlsx.close()
        self.xlsx = None

import random
from copy import deepcopy
from typing import Union, Dict, Tuple, List, TypeVar, Callable
from diffusion.utilities.data_io import SimpleExcelWriter, SimpleExcelReader
from diffusion.utilities.file_ops import mkdir, gd
from diffusion.utilities.misc import minibar, Timer, printx

# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use of 
# `T` to annotate `self`. Many methods of `Database` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Database`.
T = TypeVar('T', bound='Database')

class Database:
    def __init__(self: T, 
        db_keys: List[str], 
        xlsx_file:  str = None, 
        sheet_name: str = 'Sheet1'
    ):
        '''
        define database structure and loading functions
        '''
        def _check_if_db_keys_are_valid(db_keys):
            if not isinstance(db_keys, list): return False
            if len(db_keys) == 0: return False
            for item in db_keys:
                if not isinstance(item, str): 
                    return False
            return True
        assert _check_if_db_keys_are_valid(db_keys), \
            '* Invalid database structure given (db_keys=%s). ' \
            'You need to give a list of string to properly ' \
            'define a database structure.' % str(db_keys)

        self.data_dict: dict[str, list[str]] = {}        
        for key in db_keys:
            self.data_dict[key] = []

        if xlsx_file is not None:
            self.load_xlsx(xlsx_file, sheet_name)

    @property
    def db_keys(self):
        return self.get_db_keys()

    def get_db_keys(self) -> List[str]:
        return list(self.data_dict.keys())

    def __len__(self):
        return self.num_records()
    
    def num_records(self):
        return len(self.data_dict[self.db_keys[0]])

    
    def get_record(self: T, index) -> Dict[str, str]:
        record = {}
        for key in self.db_keys:
            record[key] = self.data_dict[key][index]
        return record
    
    def get_record_from_key_val_pair(self: T, key, value) -> Tuple[int, Union[Dict[str, str], None]]:
        '''
        Description
        -----------
        Find a record from database which satisfies record[key]==value.
        If multiple records satisfy this condition, only the first 
        record will be returned. 
        
        Returns
        -----------
        On success: Returns its record index and record item.
        On failure: Returns (-1, None) if no record is found.
        '''
        if value in self.data_dict[key]:
            index = self.data_dict[key].index(value)    
            return index, self.get_record(index)
        else:
            index = -1
            return -1, None
    
    def set_record(self: T, index: int, record: Dict[str, str]):
        if index >= self.num_records():
            raise RuntimeError('index out of range! [0~%d], got %d.' % (self.num_records()-1, index))
        rec = deepcopy(record)
        for key in self.db_keys:
            if key not in list(rec.keys()):
                rec[key] = ''
        for key in list(rec.keys()):
            val = rec[key]
            if key not in self.db_keys:
                raise RuntimeError('unknown key "%s" in record.' % key)            
            self.data_dict[key][index] = val
    
    def make_empty_record(self) -> Dict[str, str]:
        # create an empty record and return to user
        record = {}
        for key in self.db_keys:
            record[key] = ''
        return record
    
    def add_record(self: T, record: Dict[str, str]):
        '''
        Append a record to the end of the database.
        '''
        assert isinstance(record, dict), 'record should be a dict.'
        rec = deepcopy(record)
        for key in self.db_keys:
            if key not in list(rec.keys()):
                rec[key] = ''
        for key in list(rec.keys()):
            val = rec[key]
            if key not in self.db_keys:
                raise RuntimeError('unknown key "%s" in record.' % key)            
            self.data_dict[key].append(val)

    def export_xlsx(self: T, xlsx_path: str, up_freezed_rows: int = 1, left_freezed_cols: int = 0):
        '''
        Export database to excel (*.xlsx).
        '''
        worksheet_name = 'Sheet1'
        xlsx = SimpleExcelWriter( xlsx_path, worksheet_names=[worksheet_name])
        topfmt = xlsx.new_format(font_color='#FFFFFF', bg_color='#606060', bold=True)
        colfmt = xlsx.new_format(font_color='#000000', bg_color='#D9D9D9')
        errfmt = xlsx.new_format(font_color='#000000', bg_color='#FF9090')
        # write table head
        for key, i in zip(self.db_keys, [q for q in range(len(self.db_keys))] ):
            xlsx.write((0, i), key, format=topfmt, worksheet_name=worksheet_name)
        # write content
        num_records = len(self.data_dict[list(self.data_dict.keys())[0]])
        for i in range(num_records):
            for key, j in zip(self.db_keys, [q for q in range(len(self.db_keys))] ):
                content = self.data_dict[key][i]
                if isinstance(content, str) == False:
                    raise RuntimeError('invalid content type when saving xlsx file. '
                        'content type is "%s", key is "%s", value is "%s".' \
                            % (str(type(content)), key, str(content) ))
                if len(content) == 0:
                    xlsx.write((i+1, j), content, format=errfmt, worksheet_name=worksheet_name)
                elif j < left_freezed_cols:
                    xlsx.write((i+1, j), content, format=colfmt, worksheet_name=worksheet_name)
                else:
                    xlsx.write((i+1, j), content, worksheet_name=worksheet_name)
        xlsx.set_freeze(up_freezed_rows,left_freezed_cols,worksheet_name=worksheet_name)
        xlsx.set_zoom(85,worksheet_name=worksheet_name)
        xlsx.set_filter((0,0), (num_records,len(self.get_db_keys())-1), worksheet_name=worksheet_name)
        mkdir(gd(xlsx_path))
        xlsx.save_and_close()
    
    def load_xlsx(self: T, xlsx_path: str, sheet_name: str = 'Sheet1'):
        xlsx = SimpleExcelReader( xlsx_path )
        max_cols = xlsx.max_column(worksheet_name=sheet_name)
        table_map = {}
        for col in range(max_cols):
            key = xlsx.read((0, col), worksheet_name=sheet_name)
            table_map[col] = key
        max_rows = xlsx.max_row(worksheet_name=sheet_name)
        for row in range(1, max_rows):
            record = self.make_empty_record()
            for col in range(0, max_cols):
                val = xlsx.read((row, col), worksheet_name=sheet_name)
                val = '' if val is None else str(val)
                key = table_map[col]
                record[key] = str(val)
            # an external excel file may contain some other keys in record,
            # we need to remove those keys before adding them to our own database
            for key in list(record.keys()):
                if key not in self.db_keys:
                    record.pop(key)
            for key in self.db_keys:
                if key not in list(record.keys()):
                    record[key] = ''
            self.add_record(record)

    def split(self: T, ratio: Union[float, list] = 0.5) -> T:
        if isinstance(ratio, float):
            n = self.num_records()
            t = int(n * ratio)
            d1, d2 = Database(db_keys=self.db_keys), Database(db_keys=self.db_keys)
            for i in range(n):
                record = self.get_record(i)
                if i<t: d1.add_record(record)
                else: d2.add_record(record)
            d1.__class__ = type(self) # type cast to derived class
            d2.__class__ = type(self) # type cast to derived class
            return d1, d2
        elif isinstance(ratio, list):
            n = [ int(r * self.num_records()) for r in ratio ]
            s = 0
            idxs = []
            for item in n:
                s += item
                idxs.append(s)
            idxs[-1] = self.num_records()
            datasets = [Database(db_keys=self.db_keys) for _ in range(len(ratio))]
            for i in range(self.num_records()):
                for idx, j in zip(idxs, range(len(idxs))):
                    if i < idx:
                        datasets[j].add_record( self.get_record(i) )
                        break
            for dataset in datasets:
                dataset.__class__ = type(self) # type cast to derived class
            return datasets

    def shuffle(self: T, seed: int = 123456) -> T:
        new_database = deepcopy(self)
        for key in self.db_keys:
            random.Random(seed).shuffle( new_database.data_dict[key] )
        new_database.__class__ = type(self) # type cast to derived class
        return new_database
        
    def clear_key(self: T, keys: Union[List[str], str]) -> T:
        '''
        clear one or multiple keys from database (but not removed).
        '''
        k = deepcopy(keys)
        new_database = deepcopy(self)
        if isinstance(k, str):
            k = [k]
        n = new_database.num_records()
        for key in k:
            if key not in new_database.data_dict:
                raise Exception('cannot find key "%s" in database.' % key)
            new_database.data_dict[key] = [''] * n
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def remove_key(self: T, keys: Union[List[str], str]) -> T:
        '''
        purge one or multiple keys out of the database.
        '''
        new_database = deepcopy(self)
        keys_ = deepcopy(keys)
        if isinstance(keys_, str): 
            keys_ = [keys_]
        for key in keys_:
            if key not in self.db_keys:
                raise RuntimeError('key "%s" is not in database.' % key)
            new_database.data_dict.pop(key)
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def add_key(self: T, new_keys: Union[List[str], str]) -> T:
        '''
        add one or multiple keys into the database.
        '''
        new_keys_ = deepcopy(new_keys)
        new_database = deepcopy(self)
        if isinstance(new_keys, str):
            new_keys_ = [new_keys_]
        for new_key in new_keys_:
            if new_key in self.db_keys:
                raise RuntimeError('key "%s" is already in database.' % new_key)
            new_database.data_dict[new_key] = [''] * self.num_records()
        new_database.__class__ = type(self) # type cast to derived class
        return new_database
    
    def remove_by_rule(self: T, remove_func: Callable[[Dict[str,str]], bool]) -> T:
        '''
        Description
        -----------
        Remove records in database using customized rule provided by "remove_func".
        "remove_func" is a callback function provided by users and its function
        prototype should be:
        >>> def custom_rule(record: Dict[str, str]) -> bool:
        >>>     # returns True if record should be removed otherwise return False.
        >>>     ...
        >>> new_database = database.remove_by_rule(custom_rule)
        '''
        new_database = Database(db_keys=self.db_keys)
        num_records = len(self.data_dict[self.db_keys[0]])
        for record_id in range(num_records):
            record = self.get_record(record_id)
            can_remove = remove_func(record)
            if can_remove:
                continue
            else:
                new_database.add_record(record)
        new_database.__class__ = type(self) # type cast to derived class
        return new_database
    
    def keep_by_rule(self: T, keep_func: Callable[[Dict[str,str]], bool]) -> T:
        '''
        Description
        -----------
        Keep records in database using customized rule provided by "keep_func".
        "keep_func" is a callback function provided by users and its function
        prototype should be:
        >>> def custom_rule(record: Dict[str, str]) -> bool:
        >>>     # returns True if record should be kept otherwise return False.
        >>>     ...
        >>> new_database = database.keep_by_rule(custom_rule)
        '''
        new_database = Database(db_keys=self.db_keys)
        num_records = len(self.data_dict[self.db_keys[0]])
        for record_id in range(num_records):
            record = self.get_record(record_id)
            can_keep = keep_func(record)
            if not can_keep:
                continue
            else:
                new_database.add_record(record)
        new_database.__class__ = type(self) # type cast to derived class
        return new_database
    
    def update_by_rule(self: T, update_func: Callable[[Dict[str,str]], Dict[str,str]]) -> T:
        '''
        Description
        -----------
        Update records in database using customized rule provided by "update_func".
        "update_func" is a callback function provided by users and its function
        prototype should be:
        >>> def custom_rule(record: Dict[str, str]) -> Dict[str, str]:
        >>>     # returns the updated record.
        >>>     ...
        >>> new_database = database.update_by_rule(custom_rule)
        '''
        def _check_if_record_is_valid(record: dict):
            assert isinstance(record, dict), 'updated record should be a dict.'
            for k, v in record.items():
                assert isinstance(k, str) and isinstance(v, str), 'key and value should be str.'
        new_database = Database(db_keys=self.db_keys)
        for record_id in range(self.num_records()):
            record = self.get_record(record_id)
            updated_record = update_func(record)
            _check_if_record_is_valid(updated_record)
            new_database.add_record(updated_record)
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def binary_split_by_rule(self: T, split_func: Callable[[Dict[str,str]], int]) -> Tuple[T, T, T]:
        '''
        Description
        -----------
        Split database in two using customized rule provided by "split_func".
        "split_func" is a callback function provided by users and its function
        prototype should be:
        >>> def custom_rule(record: Dict[str, str]) -> int:
        >>>     # returns 1, 2, or 3.
        >>>     ...
        >>> new_database_part1, new_database_part2, remaining = database.binary_split_by_rule(custom_rule)
        '''
        new_database_part1 = Database(db_keys=self.db_keys)
        new_database_part2 = Database(db_keys=self.db_keys)
        remaining_database = Database(db_keys=self.db_keys)
        for record_id in range(self.num_records()):
            record = self.get_record(record_id)
            which_part = split_func(record)
            if which_part not in [1,2,3]:
                raise RuntimeError('%s should return 1, 2, or 3, but got %s.' % (split_func.__name__, str(which_part)))
            if which_part == 1:
                new_database_part1.add_record(record)
            elif which_part == 2:
                new_database_part2.add_record(record)
            else:
                remaining_database.add_record(record)
        new_database_part1.__class__ = type(self) # type cast to derived class
        new_database_part2.__class__ = type(self) # type cast to derived class
        remaining_database.__class__ = type(self) # type cast to derived class
        return new_database_part1, new_database_part2, remaining_database
    
    def archive_by_rule(self: T, archive_func: Callable[[Dict[str,str]], str]):
        '''
        Description
        -----------
        Archive database using customized rule.
        "archive_func" is a callback function provided by users and its function
        prototype should be:
        >>> def custom_rule(record: Dict[str, str]) -> str:
        >>>     # returns 'Success' if archive for this record succeed.
        >>>     # returns 'Failed'  if archive for this record fails.
        >>>     # returns 'Skipped' if archive for this record is skipped.
        >>>     ...
        >>> database.archive_by_rule(custom_rule)
        '''
        success, failed, skipped = 0, 0, 0
        timer = Timer()
        for record_id in range(self.num_records()):
            record = self.get_record(record_id)
            archive_state = archive_func(record)
            if   archive_state == 'Success': success += 1
            elif archive_state == 'Failed':  failed += 1
            elif archive_state == 'Skipped': skipped += 1
            else:
                raise RuntimeError('User returned unknown archive state "%s" in function "%s". '
                                   'Returned archive state should be one of "Success", "Failed", or "Skipped".' % \
                                   (str(archive_state), archive_func.__name__))
            minibar('Archiving database', a=record_id+1, b=self.num_records(), time=timer.elapsed(), 
                    last='%d success, %d failed, %d skipped.' % (success, failed, skipped))
        printx('')


    def keep_first_n(self: T, n: int) -> T:
        '''
        keep first n records in database
        '''
        new_database = Database(db_keys=self.db_keys)
        for i in range(self.num_records()):
            if i == n: break
            new_database.add_record(self.get_record(i))
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def remove_first_n(self: T, n:int) -> T:
        '''
        remove first n records in database
        '''
        new_database = Database(db_keys=self.db_keys)
        for i in range(n, self.num_records()):
            new_database.add_record(self.get_record(i))
        new_database.__class__ = type(self) # type cast to derived class
        return new_database
    
    def keep_last_n(self: T, n:int) -> T:
        '''
        keep last n records in database
        '''
        new_database = Database(db_keys=self.db_keys)
        start_ind = self.num_records() - n
        if start_ind < 0: start_ind = 0
        for i in range(start_ind, self.num_records()):
            new_database.add_record(self.get_record(i))
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def remove_last_n(self: T, n:int) -> T:
        '''
        remove last n records in database
        '''
        new_database = Database(db_keys=self.db_keys)
        for i in range(self.num_records()-n):
            new_database.add_record(self.get_record(i))
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def reverse(self: T) -> T:
        '''
        reverse all records
        '''
        new_database = Database(self.db_keys)
        for key in self.db_keys:
            new_database.data_dict[key] = deepcopy(self.data_dict[key])
            new_database.data_dict[key].reverse()
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def __add__(self: T, other: T) -> T:
        new_database = Database(db_keys=self.db_keys)
        for i in range(self.num_records()):
            new_database.add_record(self.get_record(i))
        for i in range(other.num_records()):
            new_database.add_record(other.get_record(i))
        new_database.__class__ = type(self) # type cast to derived class
        return new_database

    def __iadd__(self: T, other: T) -> T:
        for i in range(other.num_records()):
            self.add_record(other.get_record(i))
        return self

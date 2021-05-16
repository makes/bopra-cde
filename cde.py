import toml
import argparse
import re
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import nirs

def read_args():
    arg_parser = argparse.ArgumentParser(description="""
        Do CDE analysis.
        """)
    arg_parser.add_argument('config', help='name of configuration file')
    arg_parser.add_argument('--filecheck',
                            action='store_true',
                            help='perform dry run without loading data')
    arg_parser.add_argument('--generate-images',
                            action='store_true',
                            help='create visualizations')
    return arg_parser.parse_args()

ARGS = read_args()
CONFIG_FILENAME = os.path.join('./', ARGS.config)
CFG = toml.load(CONFIG_FILENAME)

CDE_DUR = CFG['cde_duration'] # seconds
CDE_DEPTH = CFG['cde_depth']  # percentage points
INTERP = CFG['interpolation'] # interpolation method: 'pad' or 'linear'

RUN_CONFIG_NAME = CFG['name']
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M")

LOGPATH = os.path.expanduser(CFG['output']['logdir'])
LOGFILE = os.path.join(LOGPATH, f'cde_{RUN_CONFIG_NAME}{TIMESTAMP}.log')

CSVPATH = os.path.expanduser(CFG['output']['csvdir'])
CSVFILE = os.path.join(CSVPATH, f'cde_{RUN_CONFIG_NAME}{TIMESTAMP}.csv')

IMGPATH = os.path.expanduser(CFG['output']['imagedir'])

CSV_SEP = CFG['output']['csv_sep']
DEC_SEP = CFG['output']['csv_decimal_sep']

DISCARD_COL = CFG['input']['amend_discard_col']

def log(line='', end='\n'):
    print(line)
    with open(LOGFILE, 'a') as f:
        f.write(str(line) + end)

class Dataset:
    def __init__(self, case_id: int, data_files: list, amend_file: str):
        self.case_id = case_id
        self.data_files = data_files
        self.amend_file = amend_file

        self.nirs_data = None
        self.amend = None

        self.data = None
        self.valid = False

        self.analysis = None

    def load_nirs(self):
        if self.data_files == None: # special handling when no raw data available
            try:
                self.nirs_data = nirs.load_csv(self.amend_file)
            except ValueError as err:
                log(f'ValueError loading NIRS data, id {self.case_id}: ', end='')
                log(err)
                self.nirs_data = None
            return

        try: # normal handling
            self.nirs_data = nirs.load_raw_csv(self.data_files[0])
        except ValueError as err:
            log(f'ValueError loading NIRS data, id {self.case_id}: ', end='')
            log(err)
            self.nirs_data = None
            return
        for f in self.data_files[1:]:
            try:
                to_join = [self.nirs_data, nirs.load_raw_csv(f)]
                self.nirs_data = pd.concat(to_join).sort_index()
            except ValueError as err:
                msg = 'ValueError joining multipart NIRS data, id '
                log(msg + f'{self.case_id}: ', end='')
                log(err)
                self.nirs_data = None
                return

    def load_amend(self):
        self.amend = pd.read_csv(self.amend_file,
                                 sep = ';',
                                 na_values = ['--', ' '],
                                 parse_dates = ['Time'])

    def aggregate(self):
        if self.nirs_data is None or self.amend is None:
            log(f"Error: data not loaded, can't aggregate id {self.case_id}")
            return
        if len(self.nirs_data.index) != len(self.amend.index):
            log(f'Error: data length mismatch: case {self.case_id}:')
            msg = f'raw: {len(self.nirs_data.index)}, '
            log(msg + f'amend: {len(self.amend.index)}')
            return
        self.data = pd.DataFrame()
        discard_col = DISCARD_COL
        if self.data_files == None:
            discard_col = 'PoorSignalQuality' # TODO: hardcoded value
        try:
            self.data['rSO2'] = self.nirs_data['rSO2']
            self.data['Mark'] = self.amend['Mark'].array
            self.data['Bad_rSO2_auto'] = self.nirs_data['Bad_rSO2_auto']
            self.data['Bad_rSO2_manual'] = self.amend[discard_col].array
        except KeyError as err:
            log(f'KeyError in case id {self.case_id}: ' + str(err))
            self.data = None
            return
        self.valid = True

    def load(self):
        self.load_nirs()
        self.load_amend()
        self.aggregate()

    def analyze(self):
        self.analysis = Analysis(self)
        self.analysis.execute()

    def get_result(self):
        return self.analysis.get_result()

    def get_csv(self):
        return self.analysis.generate_csv_line()

    def __str__(self):
        return f'{self.case_id:6}: {self.data_files}: {self.amend_file}'

class Analysis:
    def __init__(self, dataset: Dataset):
        if dataset.valid == False or dataset.data is None:
            raise ValueError("Attempt to analyze invalid dataset")
        self.dataset = dataset
        self.timestamps = dict()
        self.dropout = dict() # intervals

        self.baseline = None
        self.cde_threshold = None

        self.success = False
        self.comment = ''
        self.has_cde = False

    def execute(self):
        df = self.dataset.data
        self.dropout['auto'] = self.get_intervals('Bad_rSO2_auto')
        self.dropout['manual'] = self.get_intervals('Bad_rSO2_manual')
        # do dropout:
        df['rSO2'] = np.where(df['Bad_rSO2_manual'] != 0, np.nan, df['rSO2'])
        # resolve timestamps:
        try:
            ts, tm, te = self.resolve_timestamps()
        except AssertionError as err:
            self.comment = str(err)
            log(f'{self.dataset.case_id}: Unable to analyze - ' + str(err))
            if ARGS.generate_images == True:
                self.plot_cde(None, None)
            return False
        # compute baseline rSO2:
        self.baseline = df['rSO2'][ts:tm].mean()
        self.cde_threshold = self.baseline - CDE_DEPTH
        if self.baseline is np.nan:
            self.baseline = None
            self.comment = "Undefined baseline rSO2"
            log(f'{self.dataset.case_id}: Unable to analyze - ' + self.comment)
            if ARGS.generate_images == True:
                self.plot_cde(None, None)
            return False
        # interpolate
        df['rSO2'] = df['rSO2'].interpolate(limit_area='inside', method=INTERP)
        # find CDE
        df['rSO2_below_threshold'] = df['rSO2'] <= self.cde_threshold
        bti = self.get_intervals('rSO2_below_threshold')
        cde_ranges = []
        cd_ranges = []
        for r in bti:
            if (r[1] - r[0]).total_seconds() >= CDE_DUR:
                cde_ranges.append(r)
            else:
                cd_ranges.append(r)
        if ARGS.generate_images == True:
            self.plot_cde(cd_ranges, cde_ranges)
        if len(cde_ranges) > 0:
            self.has_cde = True
        self.success = True
        return True

    def plot_cde(self, cd_ranges, cde_ranges):
        case_id = self.dataset.case_id
        df = self.dataset.data
        fig, ax = plt.subplots(figsize=(20,6))
        ax.set_ylabel('rSO2', color='navy')
        ax.plot(df.index, df['rSO2'], color='navy', label='rSO2')
        ax.tick_params(axis='y', labelcolor='navy')
        title = f'Case {case_id}'
        if 'mark' in self.timestamps and self.baseline is not None:
            title += f' baseline={self.baseline:.1f}'
            ax.hlines(self.baseline, xmin=self.timestamps['start'], xmax=self.timestamps['mark'], color='darkgreen', linestyle='--', linewidth=2)
            title += f' threshold={self.cde_threshold:.1f}'
            ax.hlines(self.cde_threshold, xmin=self.timestamps['mark'], xmax=self.timestamps['end'], color='darkred', linestyle='--', linewidth=2)
            ax.axvline(self.timestamps['mark'], linestyle='--', linewidth=3)
        plt.title(title)
        if cd_ranges is not None:
            for r in cd_ranges:
                ax.axvspan(r[0], r[1], alpha=0.3, color='green', lw=0)
        if cde_ranges is not None:
            for r in cde_ranges:
                ax.axvspan(r[0], r[1], alpha=0.3, color='maroon', lw=0)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        plt.autoscale(enable=True, axis="x", tight=True)
        filename = f'cde_{case_id}.png'
        imgfile = os.path.join(IMGPATH, filename)
        plt.savefig(imgfile, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def generate_csv_line(self):
        success = '1' if self.success == True else '0'
        cde = '1' if self.has_cde == True else '0'
        cde = '-' if self.success == False else cde
        if self.success == False:
            baseline = '-'
        else:
            baseline = f'{self.baseline:.1f}'.replace('.', DEC_SEP)
        csv = f'{self.dataset.case_id}{CSV_SEP}{success}{CSV_SEP}'
        csv += f'{baseline}{CSV_SEP}{cde}{CSV_SEP}{self.comment}'
        return csv

    def get_result(self):
        return self.success, self.has_cde

    def resolve_timestamps(self):
        df = self.dataset.data
        try:
            self.timestamps['mark'] = df.loc[df['Mark'] == 1].index[0]
        except IndexError:
            self.timestamps['mark'] = None
            raise AssertionError("No 'Mark' found.")
        self.timestamps['start'] = df['rSO2'].first_valid_index()
        self.timestamps['end'] = df.index[-1]
        return (self.timestamps['start'],
                self.timestamps['mark'],
                self.timestamps['end'])

    def get_intervals(self, colname: str):
        """Returns start and end timestamps of intervals where column is 1."""
        df = self.dataset.data
        ranges = []
        range_start = None
        for index, row in df.iterrows():
            if row[colname] == 1 and range_start is None:
                range_start = index
            if row[colname] == 0 and range_start is not None:
                ranges.append((range_start, index))
                range_start = None
        if range_start is not None:
            ranges.append((range_start, df.index[-1]))
        return ranges


class Base:
    def __init__(self, name, data_dir, amend_dir):
        self.name = name
        if data_dir != '':
            self.data_dir = os.path.expanduser(data_dir)
        else:
            msg = f"{name}: Raw data directory not specified. "
            msg += "Using 'amend' files as NIRS data source."
            log(msg)
            self.data_dir = ''
        self.amend_dir = os.path.expanduser(amend_dir)

        # list files
        if data_dir != '':
            self.files_in_data = next(os.walk(self.data_dir))[2]
        else:
            self.files_in_data = []
        self.files_in_amend = next(os.walk(self.amend_dir))[2]

        # filter valid filenames
        reg = re.compile('nirsraaka_\d+(?:_osa\d)?\.csv')
        self.data_files = list(filter(reg.search, self.files_in_data))
        reg = re.compile('nirs_\d+_a2\.csv')
        self.amend_files = list(filter(reg.search, self.files_in_amend))

        # group multipart NIRS datafiles
        reg = re.compile('nirsraaka_\d+\.csv')
        singlepart = list(filter(reg.search, self.data_files))
        self.grouped_data_files = [[f] for f in singlepart]
        for case_id in self.get_multipart_ids():
            start = f'nirsraaka_{case_id}_osa'
            parts = [f for f in self.data_files if f.startswith(start)]
            self.grouped_data_files.append(sorted(parts))

        self.datasets = dict()

        if data_dir == '':
            # create datasets using amend files only
            reg = re.compile('nirs_(\d+)_a2\.csv')
            for amend_file in self.amend_files:
                id_search = re.search(reg, amend_file)
                if id_search is None:
                    log(f'Failed to extract case_id from filename {amend_file}.')
                    continue
                case_id = id_search.group(1)
                amend_full = os.path.join(self.amend_dir, amend_file)
                self.datasets[case_id] = Dataset(case_id, None, amend_full)
            log(f'Finished collecting files in {self.name}.')
            return

        # create datasets
        for filelist in self.grouped_data_files:
            try:
                case_id = Base.find_id(filelist)
            except Exception as err:
                log('Error (skipping file): ', end='')
                log(err)
                continue
            amend_file = self.find_amend(case_id)
            if amend_file == None:
                fpath = os.path.join(self.data_dir, filelist[0])
                log('Error (skipping file): ', end='')
                log(f'No amend file found for {fpath}')
            else:
                fl_full = [os.path.join(self.data_dir, f) for f in filelist]
                amend_full = os.path.join(self.amend_dir, amend_file)
                if case_id in self.datasets: # duplicate found
                    log('Error (skipping files): ', end='')
                    msg = f'Duplicate data found for {filelist[0]}: '
                    msg += f'{self.datasets[case_id].data_files[0]}.'
                    log(msg)
                    del self.datasets[case_id]
                else:
                    self.datasets[case_id] = Dataset(case_id, fl_full, amend_full)
        log(f'Finished collecting files in {self.name}.')

    def __str__(self):
        ret = f'Base: {self.name}\n'
        ret += f'data_dir: {self.data_dir}    amend_dir: {self.amend_dir}\n'
        ret += f'Files in data_dir:    {len(self.files_in_data):3} '
        ret += f'Files in amend_dir:    {len(self.files_in_amend):3}\n'
        ret += f'Valid data filenames: {len(self.data_files):3} '
        ret += f'Valid amend filenames: {len(self.amend_files):3}\n'
        ret += f'Datasets:             {len(self.datasets):3}'
        return ret

    def print_invalid_filenames(self):
        data_inv = [f for f in self.files_in_data if f not in self.data_files]
        amend_inv = [f for f in self.files_in_amend if f not in self.amend_files]
        if len(data_inv) > 0 or len(amend_inv) > 0:
            log(f'{self.name}: The following invalid filenames were found:')
            for s in data_inv:
                log(s)
            for s in amend_inv:
                log(s)
            log()
        else:
            log(f'{self.name}: No invalid filenames were found.\n')


    def get_multipart_ids(self):
        reg = re.compile('nirsraaka_(\d+)_osa\d?\.csv')
        multipart = list(filter(reg.search, self.data_files))
        ids = set()
        for f in multipart:
            id_search = re.search(reg, f)
            if id_search:
                ids.add(id_search.group(1))
        return ids

    def find_amend(self, case_id):
        filename = f'nirs_{case_id}_a2.csv'
        if filename in self.amend_files:
            return filename
        else:
            return None

    @classmethod
    def find_id(cls, filelist):
        if len(filelist) == 1: # not multipart
            reg = re.compile('nirsraaka_(\d+)\.csv')
            if '_osa' in filelist[0]:
                msg = f'multipart data with only one file: {filelist[0]}'
                raise Exception(msg)
        else: # multipart
            reg = re.compile('nirsraaka_(\d+)_osa\d?\.csv')
        id_search = re.search(reg, filelist[0])
        if id_search:
            return id_search.group(1)
        else:
            return None

    def purge_invalid_datasets(self):
        ids = list(self.datasets.keys())
        for case_id in ids:
            if self.datasets[case_id].valid == False:
                del self.datasets[case_id]

    def load(self):
        for case_id, dataset in self.datasets.items():
            dataset.load()
        self.purge_invalid_datasets()

    def analyze(self):
        for case_id, dataset in self.datasets.items():
            dataset.analyze()

    def get_results(self):
        failed = 0
        cde = 0
        no_cde = 0
        for case_id, dataset in self.datasets.items():
            success, has_cde = dataset.get_result()
            if success == False:
                failed += 1
                continue
            if has_cde == True:
                cde += 1
            else:
                no_cde += 1
        return failed, no_cde, cde


    def get_csv(self):
        csv = []
        for case_id, dataset in self.datasets.items():
            csv.append(self.name + CSV_SEP + dataset.get_csv())
        return '\n'.join(csv)

def main():
    starttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f'Run started at {starttime}, configuration: {RUN_CONFIG_NAME}\n')

    bases = [Base(b['name'], b['data_dir'], b['amend_dir']) for b in CFG['bases']]
    log()
    for b in bases:
        b.print_invalid_filenames()
    if ARGS.filecheck == False:
        for b in bases:
            log(f'{b.name}: Loading...')
            b.load()
            msg = f'{b.name}: successfully loaded and aggregated '
            log(msg + f'{len(b.datasets)} datasets.\n')

        csv = f'base{CSV_SEP}case_id{CSV_SEP}success{CSV_SEP}'
        csv += f'baseline{CSV_SEP}cde{CSV_SEP}comment'
        for b in bases:
            log(f'{b.name}: Analyzing...')
            b.analyze()
            log()
            csv += '\n' + b.get_csv()

        log('BASE SUMMARY:')
        log('-------------')
        for b in bases:
            log()
            log(b)

        log()
        log('SUMMARY STATISTICS:')
        log('-------------------')
        failed_tot = 0
        no_cde_tot = 0
        cde_tot = 0
        for b in bases:
            failed, no_cde, cde = b.get_results()
            failed_tot += failed
            no_cde_tot += no_cde
            cde_tot += cde
        log(f'Total cases: {failed_tot+no_cde_tot+cde_tot}')
        log(f'Analysis successful: {no_cde_tot+cde_tot}, failed: {failed_tot}')
        log(f'CDE detected: {cde_tot}, no CDE: {no_cde_tot}')
        log(f'CDE percentage: {(cde_tot * 100) / (no_cde_tot + cde_tot):.2f} %')

        with open(CSVFILE, 'w') as f:
            f.write(csv)
        log(f'\nCSV saved in {CSVFILE}')

if __name__ == "__main__":
    main()

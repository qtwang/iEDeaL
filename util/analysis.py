# coding = utf-8

import os
import csv

import numpy as np


class Collector():
    def __init__(self):
        self.fetch_all = False

        self.__epsilon = 1e-5


    def __fscore(self, beta, pre, rec):
        if pre < self.__epsilon or rec < self.__epsilon:
            return 0

        beta2 = beta ** 2
        return (1 + beta2) * pre * rec / (beta2 * pre + rec)


    def collect(self, result_rootfolder, byvalid=True, checkF = lambda folderpath : True, fetch_all: bool = False, beta: float = None):
        if beta is not None:
            assert beta > 0

        assert os.path.exists(result_rootfolder)
        self.fetch_all = fetch_all

        self.results = {}
        self.trial_counts = {}

        for patient_id in os.listdir(result_rootfolder):
            patient_folderpath = os.path.join(result_rootfolder, patient_id)

            if os.path.isdir(patient_folderpath):
                for setting in os.listdir(patient_folderpath):
                    setting_folderpath = os.path.join(patient_folderpath, setting)

                    if os.path.isdir(setting_folderpath):
                        for tune in os.listdir(setting_folderpath):
                            tune_folderpath = os.path.join(setting_folderpath, tune)
                            
                            if os.path.isdir(tune_folderpath):
                                if set(['model.pickle', 'train.log']).issubset(os.listdir(tune_folderpath)) and checkF(tune_folderpath):
                                    setting_segments = setting.split('-')
                                    if 'n' in setting_segments[-1]:
                                        beta_fetched = float(setting_segments[-2])
                                    else:
                                        beta_fetched = float(setting_segments[-1])

                                    log_filepath = os.path.join(tune_folderpath, 'train.log')
                                    
                                    finished = False
                                    with open(os.path.join(log_filepath), 'r') as fin:
                                        for line in fin:
                                            if 'training failed' in line or 'training early stopped' in line or 'training finished' in line:
                                                finished = True
                                                break

                                    if finished:
                                        train_results = []
                                        valid_results = []

                                        with open(log_filepath, 'r') as fin:
                                            for line in fin:
                                                if 'l=' in line and 'f' in line and ('pre=' in line or 'p=' in line) and ('rec=' in line or 'r=' in line):
                                                    segments = line.split(':')[-1].split(', ')

                                                    loss, fscore, pre, rec = [float(segments[i].split('=')[1]) for i in range(4)]

                                                    if beta is not None and beta != beta_fetched:
                                                        fscore = self.__fscore(beta, pre, rec)

                                                    train_results.append([loss, fscore, pre, rec])
                                                elif 'f' in line and ('pre=' in line or 'p=' in line) and ('rec=' in line or 'r=' in line):
                                                    segments = line.split(':')[-1].split(', ')

                                                    fscore, pre, rec = [float(segments[i].split('=')[1]) for i in range(3)]

                                                    if beta is not None and beta != beta_fetched:
                                                        fscore = self.__fscore(beta, pre, rec)

                                                    valid_results.append([fscore, pre, rec])

                                        if byvalid:
                                            results2sortby = valid_results
                                            positions2compare = [0, 0]
                                        else:
                                            results2sortby = train_results
                                            positions2compare = [1, 1]

                                        if len(results2sortby) > 0 and not np.isnan(results2sortby[-1][positions2compare[1]]):
                                            if patient_id not in self.results:
                                                self.results[patient_id] = {}
                                                self.trial_counts[patient_id] = {}
                                                
                                            if setting not in self.results[patient_id]:
                                                self.results[patient_id][setting] = []
                                                self.trial_counts[patient_id][setting] = 0

                                            if self.fetch_all:
                                                self.results[patient_id][setting].append([valid_results[-1], train_results[-1], os.path.join(tune_folderpath, 'model.pickle')])
                                            elif len(self.results[patient_id][setting]) == 0 or results2sortby[-1][positions2compare[1]] > self.results[patient_id][setting][positions2compare[0]][positions2compare[1]]:
                                                    self.results[patient_id][setting] = [valid_results[-1], train_results[-1], os.path.join(tune_folderpath, 'model.pickle')]

                                            self.trial_counts[patient_id][setting] += 1


    def dump(self, dump_filepath):
        with open(dump_filepath, 'w') as csv_fout:
            csv_writer = csv.writer(csv_fout, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['source_patient', 'target_patient', 'setting', 'f1', 'precision', 'recall'])

            for patient_id, result4setting in self.results.items():
                for setting, result in result4setting.items():
                    csv_writer.writerow([patient_id, patient_id, setting, *result[0]])

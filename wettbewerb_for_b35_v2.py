# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Funktionen zum Laden und Speichern der Dateien
"""
__author__ = "Maurice Rohr und Dirk Schweickard"
import csv
import matplotlib.pyplot as plt
import mne
from mne.filter import  notch_filter , filter_data
from mne import create_info , rename_channels, set_bipolar_reference
from mne.io import RawArray
from mne.channels import make_1020_channel_selections
from scipy import signal as sig
from scipy.fft import fft
import ruptures as rpt
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable 
from torch.utils.data import TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding,TSNE
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import umap
from scipy.signal import stft
from sklearn.decomposition import FastICA
import pywt
from scipy.interpolate import interp1d
import math

from typing import List, Tuple, Dict, Any
import csv
import scipy.io as sio
import numpy as np
import os
from wettbewerb import load_references, get_3montages, get_6montages
from sklearn.decomposition import FastICA
import pywt
from utils import progressbar
from tqdm import tqdm

def extract_wavelet_features(coeffs):
    
    features = []
    
    for i in range(len(coeffs)):
        # 提取每个小波子带的统计特征
        mean_value = np.mean(coeffs[i])
        std_dev = np.std(coeffs[i])
        skewness = np.mean(np.power((coeffs[i] - mean_value) / std_dev, 3))
        kurtosis = np.mean(np.power((coeffs[i] - mean_value) / std_dev, 4)) - 3
        
        # 将特征组合成一个向量
        feature_vector = [std_dev, skewness, kurtosis]
        features.append(feature_vector)
    
    return features

def flatten_comprehension(matrix):
        return [item for row in matrix for item in row]

def apply_wavelet_transform(signal, wavelet='db1', level=4):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs

def upsample_eeg(original_eeg, target_length):

    original_time = np.arange(len(original_eeg))
    target_time = np.linspace(0, len(original_eeg) - 1, target_length)

    interpolator = interp1d(original_time, original_eeg, kind='linear', fill_value='extrapolate')
    upsampled_eeg = interpolator(target_time)

    return upsampled_eeg

"""
    item 1: channel
    item 2: data
    item 3: fs
"""

def re_reference(item1, item2, item3):
    # 重参考
    raw = []
    info = create_info(ch_names=item1,
                           ch_types=['eeg'] * len(item1),
                           sfreq=item3)
    raw_single = RawArray(data=item2,info=info)

    # wrong reference correction:
    rename_dict = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
    rename_channels(raw_single.info, rename_dict)
    raw_single.set_montage('easycap-M1')

    # Mastoids Reference:
    rename_dict2 = {'T7':'M1','T8':'M2'}
    rename_channels(raw_single.info, rename_dict2)

    # Set mastroid channels:该mne.channels.make_1020_channel_selections函数用于根据用于 EEG 电极放置的国际 10-20 系统
    # 创建按头部区域分类的通道索引字典。返回的字典中的键对应于不同的区域，如“左”、“右”、“前”、“中”等。
    ch_indices = make_1020_channel_selections(raw_single.info)
    # set channel names as library: Left head and right head?
    channel_names = {
        'Left': [raw_single.ch_names[ch] for ch in ch_indices['Left']],
        'Right': [raw_single.ch_names[ch] for ch in ch_indices['Right']]
    }
    # Remove M1,M2 from indices, prevent "M1-M1","M2-M2"
    channel_names['Left'].remove('M1')
    channel_names['Right'].remove('M2')

    # left half:
    set_bipolar_reference(
        raw_single,
        anode=channel_names['Left'],
        cathode=['M1']*len(channel_names['Left']),
        copy=False
    )
    # right half:
    set_bipolar_reference(
        raw_single,
        anode=channel_names['Right'],
        cathode=['M2']*len(channel_names['Right']),
        copy=False
    )
    Raw_single = raw_single.get_data().tolist()
    
    return raw_single.ch_names, Raw_single


def Exception_data_handling(Rawsingle):
    # threshold = 8 * 10**-4  # 定义阈值
    modified_Raw_single = []
    for array in Rawsingle:
        # 将列表转换为 NumPy 数组
        np_array = np.array(array)
        # # 将大于阈值的元素替换为阈值
        # np_array[np_array > threshold] = threshold
        # np_array[np_array < -1*threshold] = -1*threshold
        # 将修改后的 NumPy 数组添加到子列表
        modified_Raw_single.append(np_array)
    return modified_Raw_single


def band_pass_features(signal, _fs):
    # feature = []
    # 进行傅里叶变换
    fft_result = fft(signal)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/_fs)

    # 定义频率带范围
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 20)
    # gamma_band = (30, 70)

    # 在频率域上分离不同频率带的信号
    delta_power = np.sum(np.abs(fft_result[(fft_freqs >= delta_band[0]) & (fft_freqs <= delta_band[1])])) ** 2
    theta_power = np.sum(np.abs(fft_result[(fft_freqs >= theta_band[0]) & (fft_freqs <= theta_band[1])])) ** 2
    alpha_power = np.sum(np.abs(fft_result[(fft_freqs >= alpha_band[0]) & (fft_freqs <= alpha_band[1])])) ** 2
    beta_power = np.sum(np.abs(fft_result[(fft_freqs >= beta_band[0]) & (fft_freqs <= beta_band[1])])) ** 2
    # gamma_power = np.sum(np.abs(fft_result[(fft_freqs >= gamma_band[0]) & (fft_freqs <= gamma_band[1])])) ** 2

    # 计算Mean Spectrum Amplitude
    delta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= delta_band[0]) & (fft_freqs <= delta_band[1])]))
    theta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= theta_band[0]) & (fft_freqs <= theta_band[1])]))
    alpha_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= alpha_band[0]) & (fft_freqs <= alpha_band[1])]))
    beta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= beta_band[0]) & (fft_freqs <= beta_band[1])]))
    # gamma_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= gamma_band[0]) & (fft_freqs <= gamma_band[1])]))
    
    # tmp_feature.append([delta_power, theta_power, alpha_power, beta_power, gamma_power, delta_mean_amp, theta_mean_amp, alpha_mean_amp, beta_mean_amp, gamma_mean_amp])
    
    return [delta_power, theta_power, alpha_power, beta_power, delta_mean_amp, theta_mean_amp, alpha_mean_amp, beta_mean_amp]
    

### Achtung! Diese Funktion nicht veraendern.
def load_references_transformer(start, end, folder: str = '../training'):
    
    """
    Liest Referenzdaten aus .mat (Messdaten) und .csv (Label) Dateien ein.
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.

    Returns
    -------
    ids : List[str]
        Liste von ID der Aufnahmen
    channels : List[List[str]]
        Liste der vorhandenen Kanäle per Aufnahme
    data :  List[ndarray]
        Liste der Daten pro Aufnahme
    sampling_frequencies : List[float]
        Liste der Sampling-Frequenzen.
    reference_systems : List[str]
        Liste der Referenzsysteme. "LE", "AR", "Sz" (Zusatz-Information)
    """
    
# Initialisiere Listen ids, channels, data, sampling_frequencies, refernece_systems und eeg_labels
    ids: List[str] = []
    channels: List[List[str]] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    feature = []
    seizureRatios = []
    Times = []
    #===================
    Channels = []
    #===================
    
 # Erzeuge Datensatz aus Ordner und fülle Listen mit Daten
    dataset = EEGDataset(folder)
    # for item in progressbar(dataset, 'Loading: ', 40):
    
    for i in tqdm(range(start, end)):
    # for item in tqdm(dataset):
        item = dataset.__getitem__(i)
        ids.append(item[0])
        channels.append(item[1])
        # fs = item[3]
        # dataLength = len(item[2][0])
        onsetIndex = int(item[5][1]*item[3])
        offsetIndex = int(item[5][2]*item[3])
        maxGap = offsetIndex - onsetIndex
        
        expTime = len(item[2][0]) / item[3]
        duration = item[5][2] - item[5][1]
        seizureRatio = duration / expTime
    
        newData = []
        
        if seizureRatio > 0.1 and seizureRatio < 0.3:
            
            gap = np.random.randint(1, maxGap - 1)
            
            if gap/item[3] < item[5][1] and (maxGap - gap)/item[3] < expTime - item[5][2]:
                newStartIndex = onsetIndex - gap 
                timeGap = newStartIndex / item[3]
                newEndIndex = offsetIndex + maxGap - gap
                for i in range(len(item[2])):
                    tmp = item[2][i][newStartIndex:newEndIndex]
                    newData.append(tmp)
                print(f'old time label are {item[5][1]} and {item[5][2]}')
                Lable = (item[5][0], item[5][1] - timeGap, item[5][2] - timeGap)
                print(f'new time label are { Lable[1]} and { Lable[2]}')
                
            elif gap/item[3] < expTime - item[5][2] and (maxGap - gap)/item[3] < item[5][1]:
                newStartIndex = onsetIndex - maxGap + gap
                timeGap = newStartIndex / item[3]
                newEndIndex = offsetIndex + gap
                for i in range(len(item[2])):

                    tmp = item[2][i][newStartIndex:newEndIndex]
                    newData.append(tmp)
                print(f'old time label are {item[5][1]} and {item[5][2]}')
                Lable = (item[5][0], item[5][1] - timeGap, item[5][2] - timeGap)
                print(f'new time label are { Lable[1]} and { Lable[2]}')

            else:
                newData = item[2]
                Lable = item[5]
                
            expTime = len(newData[0]) / item[3]
            duration = Lable[2] - Lable[1]
            seizureRatio = duration / expTime
            seizureRatios.append(seizureRatio)
            
        else:
            newData = item[2]
            Lable = item[5]
            if seizureRatio > 0:
                seizureRatios.append(seizureRatio)
 #===================================================================================================================================================
        # 重参考
        # Raw_single: data
        channel, Raw_single = re_reference(item[1], newData, item[3] )
        Channels.append(channel)
        
        #剪除异常数据
        modified_Raw_single = Exception_data_handling(Raw_single)
        _montage, _montage_data, _is_missing = Get_6Montages(channel, modified_Raw_single)
    
        # print('_montage_data: ', _montage_data[0])
        
 # 使用独立成分分析（ICA）对信号进行盲源分离
        # ica = FastICA(n_components=len(_montage), random_state = 42)
        # signals_ica = ica.fit_transform(_montage_data.T).T
    
        signal_filters = []

        for j, signal_name in enumerate(_montage):
            _montage_data[j] = (_montage_data[j] - min(_montage_data[j])) / (max(_montage_data[j]) - min(_montage_data[j]))

            #陷波滤波
            # signal = notch_filter(x=_montage_data[j], Fs=item[3], freqs=np.array([60.]), n_jobs=2, verbose=False)
     
             # 带通滤波只要0.5到20Hz
            signal_filter = filter_data(data=_montage_data[j], sfreq=item[3], l_freq=0.5, h_freq = 20, method='iir', n_jobs=5, verbose=False)

            signal_filters.append(signal_filter)
            
#         #==================================================================================================================================================
            
#             #小波滤波
#             output = apply_wavelet_transform(signal_filter)
#             signal_filter1 = upsample_eeg(output[0], l)
#             idx_feature.append(signal_filter1)
#             signal_filter2 = upsample_eeg(output[1], l)
#             idx_feature.append(signal_filter2)
#             signal_filter3 = upsample_eeg(output[2], l)
#             idx_feature.append(signal_filter3)
#             signal_filter4 = upsample_eeg(output[3], l)
#             idx_feature.append(signal_filter4)
#             signal_filter5 = upsample_eeg(output[4], l)
#             idx_feature.append(signal_filter5)
             
#         # idx_feature是[30,sequence_long]型的list
#         #nperseg 为每个窗口的长度，noverlap 为窗口之间的重叠部分长度
        
        
        fs = np.abs(item[3])  # 采样频率
        
        """
        parameters vanishing
        
        1, , 500
        
        90 * 119
        """
        
        nperseg = fs * 10  # 10 秒的窗口
        noverlap = fs * 0  # 1 秒的步长

        # 存储每个窗口的平均谱幅和功率波密度
        result_array = []

        for wave in signal_filters:
            # 对每个波进行短时傅里叶变换，Zxx:(bins, 时间窗口数), times：(时间窗口数,)。
            _, times, Zxx = stft(wave, fs=fs, nperseg=nperseg, noverlap=noverlap)
            
            # 计算每个时间窗口的平均谱幅和功率波密度,形状分别：(时间窗口数，1)
            avg_spectral_amplitude = np.mean(np.abs(Zxx), axis=0)
            power_spectral_density = np.mean(np.abs(Zxx) ** 2, axis=0)
            std_spectral_amplitude = np.std(np.abs(Zxx), axis=0)
            
            # np.stack([avg_spectral_amplitude,power_spectral_density,std_spectral_amplitude], axis=1).shape)形状：(时间窗口数，3)
            result_array.append(np.stack([avg_spectral_amplitude, power_spectral_density, std_spectral_amplitude], axis=1)) 
            
        Times.append(times)   
        result_array = torch.Tensor(np.concatenate(result_array, axis=1))
        feature.append(result_array)
        sampling_frequencies.append(item[3])
        reference_systems.append(item[4])
        eeg_labels.append(Lable)
    
    data_tensor = nn.utils.rnn.pad_sequence(feature, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(seq) for seq in feature])
        
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ids)))
    return data_tensor, lengths, Times, ids, channels, sampling_frequencies, reference_systems, eeg_labels, seizureRatios

    
### Achtung! Diese Funktion nicht veraendern.
def load_references(folder: str = '../training') -> Tuple[List[str], List[List[str]],
                                                          List[np.ndarray],  List[float],
                                                          List[str], List[Tuple[bool,float,float]]]:
    """
    Liest Referenzdaten aus .mat (Messdaten) und .csv (Label) Dateien ein.
    Parameters
    ----------
    folder : str, optional
        Ort der Trainingsdaten. Default Wert '../training'.

    Returns
    -------
    ids : List[str]
        Liste von ID der Aufnahmen
    channels : List[List[str]]
        Liste der vorhandenen Kanäle per Aufnahme
    data :  List[ndarray]
        Liste der Daten pro Aufnahme
    sampling_frequencies : List[float]
        Liste der Sampling-Frequenzen.
    reference_systems : List[str]
        Liste der Referenzsysteme. "LE", "AR", "Sz" (Zusatz-Information)
    """
    
# Initialisiere Listen ids, channels, data, sampling_frequencies, refernece_systems und eeg_labels
    ids: List[str] = []
    channels: List[List[str]] = []
    new_data: List[np.ndarray] = []
    sampling_frequencies: List[float] = []
    reference_systems: List[str] = []
    eeg_labels: List[Tuple[bool,float,float]] = []
    features = []
    numNaN = 0
    #===================
    Channels=[]
    #===================
    
 # Erzeuge Datensatz aus Ordner und fülle Listen mit Daten
    dataset = EEGDataset(folder)
    for item in progressbar(dataset, 'Loading: ', 40):
        ids.append(item[0])
        channels.append(item[1])
        #===================================================================================================================================================
        # 重参考
        channel, Raw_single = re_reference(item[1], item[2], item[3])
        Channels.append(channel)
       
        # #剪除异常数据
        modified_Raw_single = Exception_data_handling(Raw_single)
        
        _montage, _montage_data, _is_missing = Get_6Montages(channel, modified_Raw_single)
        
        tmp_feature = []
        
        for j, signal_name in enumerate(_montage):


            signal = _montage_data[j]
            signal_filter = mne.filter.filter_data(data=signal, sfreq=item[3], l_freq=0.5, h_freq=40.0, n_jobs=5, verbose=False)
            
            # print(len(signal)/_fs)# 计算时间
            
            # 进行傅里叶变换
            fft_result = fft(signal_filter)
            fft_freqs = np.fft.fftfreq(len(fft_result), 1/item[3])

            # 定义频率带范围
            delta_band = (0.5, 4)
            theta_band = (4, 8)
            alpha_band = (8, 13)
            beta_band = (13, 30)
            gamma_band = (30, 40)

            # 在频率域上分离不同频率带的信号
            delta_power = np.sum(np.abs(fft_result[(fft_freqs >= delta_band[0]) & (fft_freqs <= delta_band[1])])) ** 2
            theta_power = np.sum(np.abs(fft_result[(fft_freqs >= theta_band[0]) & (fft_freqs <= theta_band[1])])) ** 2
            alpha_power = np.sum(np.abs(fft_result[(fft_freqs >= alpha_band[0]) & (fft_freqs <= alpha_band[1])])) ** 2
            beta_power = np.sum(np.abs(fft_result[(fft_freqs >= beta_band[0]) & (fft_freqs <= beta_band[1])])) ** 2
            gamma_power = np.sum(np.abs(fft_result[(fft_freqs >= gamma_band[0]) & (fft_freqs <= gamma_band[1])])) ** 2

            # 计算Mean Spectrum Amplitude
            delta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= delta_band[0]) & (fft_freqs <= delta_band[1])]))
            theta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= theta_band[0]) & (fft_freqs <= theta_band[1])]))
            alpha_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= alpha_band[0]) & (fft_freqs <= alpha_band[1])]))
            beta_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= beta_band[0]) & (fft_freqs <= beta_band[1])]))
            gamma_mean_amp = np.mean(np.abs(fft_result[(fft_freqs >= gamma_band[0]) & (fft_freqs <= gamma_band[1])]))
            
            tmp_feature.append([delta_power, theta_power, alpha_power, beta_power, gamma_power, delta_mean_amp, theta_mean_amp, alpha_mean_amp, beta_mean_amp, gamma_mean_amp])
 
        tmp_feature_array = np.array(tmp_feature)
        tmp_feature_flattened = tmp_feature_array.flatten()
        features.append(tmp_feature_flattened)
        tmp_feature = flatten_comprehension(tmp_feature)
        sampling_frequencies.append(item[3])
        reference_systems.append(item[4])
        eeg_labels.append(item[5])
        
    # Zeige an wie viele Daten geladen wurden
    print("{}\t Dateien wurden geladen.".format(len(ids)))
    return ids, channels, features, sampling_frequencies, reference_systems, eeg_labels

### Achtung! Diese Klasse nicht veraendern.
class EEGDataset:
    def __init__(self,folder:str) -> None:
        """Diese Klasse stellt einen EEG Datensatz dar.
        
        Verwendung:
            Erzeuge einen neuen Datensatz (ohne alle Daten zu laden) mit
            dataset = EEGDataset("../training/")
            len(dataset) # gibt Größe des Datensatzes zurück
            dataset[0] # gibt erstes Element aus Datensatz zurück bestehend aus (id, channels, data, sampling_frequency, reference_system, eeg_label)
            it = iter(dataset) # gibt einen iterator zurück auf den Datensatz,
            next(it) # gibt nächstes Element zurück bis alle Daten einmal geholt wurden
            for item in dataset: # iteriert einmal über den gesamten Datensatz
                (id, channels, data, sampling_frequency, reference_system, eeg_label) = item
                # Berechnung

        Args:
            folder (str): Ordner in dem der Datensatz bestehend aus .mat-Dateien und einer REFERENCE.csv Datei liegt
        """
        assert isinstance(folder, str), "Parameter folder muss ein string sein aber {} gegeben".format(type(folder))
        assert os.path.exists(folder), 'Parameter folder existiert nicht!'
        # Initialisiere Listen für ids und labels
        self._folder = folder
        self._ids: List[str] = []
        self._eeg_labels: List[Tuple[bool,float,float]] = []
        # Lade references Datei
        with open(os.path.join(folder, 'REFERENCE.csv')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Iteriere über jede Zeile
            for row in csv_reader:
                self._ids.append(row[0])
                self._eeg_labels.append((int(row[1]),float(row[2]),float(row[3])))
    
    def __len__(self):
        return len(self._ids)
    
    def __getitem__(self,idx) -> Tuple[str, List[str],
                                    np.ndarray,  float,
                                    str, Tuple[bool,float,float]]:
        #Lade Matlab-Datei
        eeg_data = sio.loadmat(os.path.join(self._folder, self._ids[idx] + '.mat'),simplify_cells=True)
        ch_names = eeg_data.get('channels')
        channels = [x.strip(' ') for x in ch_names] 
        data = eeg_data.get('data')
        sampling_frequency = eeg_data.get('fs')
        reference_system = eeg_data.get('reference_system')
        return (self._ids[idx],channels,data,sampling_frequency,reference_system,self._eeg_labels[idx])
    
    def get_labels(self):
        return self._eeg_labels
    

    

### Achtung! Diese Funktion nicht veraendern.
#predictions = {"id":id,"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
#                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
#                   "offset_confidence":offset_confidence}
def save_predictions(predictions: List[Dict[str,Any]], folder: str=None) -> None:
    """
    Funktion speichert the gegebenen predictions in eine CSV-Datei mit dem name PREDICTIONS.csv. 
    Alle Optionalen Vorherhsagen werden mit Standardwerten ersetzt.
    Parameters
    ----------
    predictions : List[Dict[str,Any]]
        Liste aus dictionaries. Jedes Dictionary enthält die Felder "id","seizure_present",
                "seizure_confidence" (optional),"onset","onset_confidence" (optional),
                "offset" (optional),"offset_confidence" (optional)
	folder : str
		Speicherort der predictions
    Returns
    -------
    None.

    """    
	# Check Parameter
    assert isinstance(predictions, list), \
        "Parameter predictions muss eine Liste sein aber {} gegeben.".format(type(predictions))
    assert len(predictions) > 0, 'Parameter predictions muss eine nicht leere Liste sein.'
    assert isinstance(predictions[0], dict), \
        "Elemente der Liste predictions muss ein Dictionary sein aber {} gegeben.".format(type(predictions[0]))
    assert "id" in predictions[0], \
        "Prädiktionen müssen eine ID besitzen, aber Key in Dictionary nicht vorhanden"
	
    if folder==None:
        file = "PREDICTIONS.csv"
    else:
        file = os.path.join(folder, "PREDICTIONS.csv")
    # Check ob Datei schon existiert wenn ja loesche Datei
    if os.path.exists(file):
        os.remove(file)

    with open(file, mode='w', newline='') as predictions_file:

        # Init CSV writer um Datei zu beschreiben
        predictions_writer = csv.writer(predictions_file, delimiter=',')
        # Iteriere über jede prediction
        header=["id","seizure_present","seizure_confidence","onset","onset_confidence","offset","offset_confidence"]
        predictions_writer.writerow(header)
        for prediction in predictions:
            _id = prediction["id"]
            _seizure_present = prediction["seizure_present"]
            _seizure_confidence = prediction.get("seizure_confidence",1.0) 
            _onset = prediction["onset"]
            _onset_confidence = prediction.get("onset_confidence",1.0) 
            _offset = prediction.get("offset",999999.0)
            _offset_confidence = prediction.get("offset_confidence",0.0)
            predictions_writer.writerow([_id,_seizure_present,_seizure_confidence,_onset,_onset_confidence,_offset,_offset_confidence])
        # Gebe Info aus wie viele labels (predictions) gespeichert werden
        print("{}\t Labels wurden geschrieben.".format(len(predictions)))
        

def get_3montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 3 Montagen Fp1-F3, Fp2-F4, C3-P3 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F3", "Fp2-F4", "C3-P3"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """   
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([3,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)


def get_6montages(channels: List[str], data: np.ndarray) -> Tuple[List[str],np.ndarray,bool]:
    """
    Funktion berechnet die 6 Montagen Fp1-F3, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F3", "Fp2-F4", "C3-P3", "F3-C3", "F4-C4", "C4-P4"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """  
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([6,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3-C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4-C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[5,:] = data[channels.index('C4-P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:         
        try:
            montage_data[0,:] = data[channels.index('Fp1')] - data[channels.index('F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2')] - data[channels.index('F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3')] - data[channels.index('P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3')] - data[channels.index('C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4')] - data[channels.index('C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[5,:] = data[channels.index('C4')] - data[channels.index('P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

def Get_6Montages(channels, data: np.ndarray):
    """
    Funktion berechnet die 6 Montagen Fp1-F3, Fp2-F4, C3-P3, F3-C3, F4-C4, C4-P4 aus den gegebenen Ableitungen (Montagen)
    zur selben Referenzelektrode. Falls nicht alle nötigen Elektroden vorhanden sind, wird das entsprechende Signal durch 0 ersetzt. 
    ----------
    channels : List[str]
        Namen der Kanäle z.B. Fp1, Cz, C3
	data : ndarray
		Daten der Kanäle
    Returns
    -------
    montages : List[str]
        Namen der Montagen ["Fp1-F3", "Fp2-F4", "C3-P3", "F3-C3", "F4-C4", "C4-P4"]
    montage_data : ndarray
        Daten der Montagen
    montage_missing : bool
        1 , falls eine oder mehr Montagen fehlt, sonst 0

    """  
    montages = []
    _,m = np.shape(data)
    montage_data = np.zeros([6,m])
    montage_missing = 0
    if '-' in channels:
        try:
            montage_data[0,:] = data[channels.index('Fp1-F3')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-F4')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[2,:] = data[channels.index('C3-P3')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3-C3')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4-C4')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')        
        try:
            montage_data[5,:] = data[channels.index('C4-P4')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)

    else:         
        try:
            montage_data[0,:] = data[channels.index('Fp1-M1')] - data[channels.index('F3-M1')]
            montages.append('Fp1-F3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[1,:] = data[channels.index('Fp2-M2')] - data[channels.index('F4-M2')]
            montages.append('Fp2-F4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[2,:] = data[channels.index('C3-M1')] - data[channels.index('P3-M1')]
            montages.append('C3-P3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[3,:] = data[channels.index('F3-M1')] - data[channels.index('C3-M1')]
            montages.append('F3-C3')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[4,:] = data[channels.index('F4-M2')] - data[channels.index('C4-M2')]
            montages.append('F4-C4')
        except:
            montage_missing = 1
            montages.append('error')
        try:
            montage_data[5,:] = data[channels.index('C4-M2')] - data[channels.index('P4-M2')]
            montages.append('C4-P4')
        except:
            montage_missing = 1
            montages.append('error')

        return (montages,montage_data,montage_missing)
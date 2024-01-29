from wettbewerb import get_3montages, get_6montages
import numpy as np
import mne
from scipy import signal as sig
from scipy.fft import fft

def pre_processing(data, channels, fs, interval):

    features = []
    _montage, _montage_data, _is_missing = get_6montages(channels, data)
    segmented_montage_data = data_segmentation(_montage_data, fs, interval)
    segmented_montage_data = np.asarray(segmented_montage_data)
    transposed_segmented_montage_data = np.transpose(segmented_montage_data, (1, 0, 2))
    
    for element in transposed_segmented_montage_data:

        element_feature_flattened = band_filter(element)
        features.append(element_feature_flattened, fs)

    return features


def data_segmentation(_montage_data, fs, interval):
    
    segmented_montage_data = []

    for i in range(len(_montage_data)):
 
        num_seg = (len(_montage_data[0]) / fs) / interval + 1
        tmp_montage_i = []

        start = 0
        end = num_seg * fs

        for j in range(num_seg):

            if start < len(_montage_data[0]) and end < len(_montage_data[0]):

                tmp_montage_i.append(_montage_data[i][start:end])
                start += num_seg * fs
                end += num_seg * fs

            else:

                tmp_montage_i.append(_montage_data[i][start:end])

        segmented_montage_data.append(tmp_montage_i)

    return segmented_montage_data


def re_label(labels, num_seg, interval):

    if labels[0] == 0:

        new_label = np.zeros(num_seg)

    else:

        new_label = []
        t = 0
        for i in range(num_seg):
            if t < labels[1] - 5:
                new_label.append(0)
                t+=interval

            elif t < labels[2] + interval/2:
                new_label.append(1)
                t+=interval

            else:
                new_label.append(0)
                t+=interval
    
    return new_label


def band_filter(element, fs):
     
        feature = []
        for item in element:

            signal_filter = mne.filter.filter_data(data=item, sfreq=fs, l_freq=0.5, h_freq=50.0, n_jobs=2, verbose=False)
            
            # 进行傅里叶变换
            fft_result = fft(signal_filter)
            fft_freqs = np.fft.fftfreq(len(fft_result), 1/fs)

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
            
            feature.append([delta_power, theta_power, alpha_power, beta_power, gamma_power, delta_mean_amp, theta_mean_amp, alpha_mean_amp, beta_mean_amp, gamma_mean_amp])

        element_feature = np.array(feature)
        element_feature_flattened = element_feature.flatten()

        return element_feature_flattened
        
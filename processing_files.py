import os
import time
import numpy as np
import pickle
import processing


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def processing_for_training(train_A_dir, cache_folder):
    num_mcep      = 128
    sampling_rate = 16000
    frame_period  = 10.0

    print("Starting to prepocess data.......")
    start_time = time.time()

    wavs_A , labels  = processing.load_wavs(wav_dir=train_A_dir, sr=sampling_rate)

    open(os.path.join(cache_folder,'texts.txt'),'w',encoding='utf8').write('\n'.join([lbl.replace('\n','') for lbl in labels]))

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps = processing.world_encode_data(
        wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean, log_f0s_std = processing.logf0_statistics(f0s=f0s_A)

    print("Log Pitch A")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean, log_f0s_std))


    coded_sps_A_transposed = processing.transpose_in_list(lst=coded_sps)


    coded_sps_norm, coded_sps_A_mean, coded_sps_A_std = processing.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)


    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    np.savez(os.path.join(cache_folder, 'logf0s_normalization.npz'),
             mean = log_f0s_mean,
             std  = log_f0s_std)

    np.savez(os.path.join(cache_folder, 'mcep_normalization.npz'),
             mean=coded_sps_A_mean,
             std=coded_sps_A_std)

    save_pickle(variable=coded_sps_norm,
                fileName=os.path.join(cache_folder, "coded_sps_norm.pickle"))
    
    open(os.path.join(cache_folder,'texts.txt'),'w',encoding='utf8').write('\n'.join([lbl.replace('\n','') for lbl in labels]))

    end_time = time.time()
    print("processinging finsihed!! see your directory ../cache for cached processinged data")

    print("Time taken for processinging {:.4f} seconds".format(
        end_time - start_time))


if __name__ == '__main__':
    processing_for_training('./data', 'cache')
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2019  Northwestern Polytechnical University (author: Yihui Fu)

import torch
import soundfile as sf
import numpy as np
import scipy as sp
import torch.utils.data as tud
from torch.utils.data import DataLoader, Dataset
import os
import sys
import multiprocessing as mp 
import scipy.io as sio
import time
class DataReader(object):
    def __init__(self, file_name, win_len=400, win_inc=100,left_context=0,right_context=0, fft_len=512, window_type='hamming', sample_rate=16000):
        self.left_context = left_context
        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.window = {
                        'hamming':np.hamming(self.win_len)/1.2607934,
                        'none':np.ones(self.win_len)
                      }[window_type]
        self.file_list = []
        parse_scp(file_name, self.file_list)

    def extract_feature(self, path):
        path = path['path']
        utt_id = path.split('/')[-1]
        
        data = audioread(path)
        inputs = enframe(data, self.window, self.win_len,self.win_inc)
        inputs = np.fft.rfft(inputs, n=self.fft_len)
        sinputs = splice_feats(np.abs(inputs).astype(np.float32), left=self.left_context, right=self.left_context)
       
        length, dims = sinputs.shape
        sinputs = np.reshape(sinputs, [1, length, dims])
        nsamples = data.shape[0]
        return sinputs, [length], np.angle(inputs), utt_id, nsamples

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

def activelev(data):
    max_val = np.max(np.abs(data))
    if max_val == 0:
        return data
    else:
        return data / max_val
        
def enframe(data, window, win_len, inc):
    data_len = data.shape[0] 
    if data_len <= win_len:
        nf = 1
    else:
        nf = int((data_len-win_len+inc)/inc)
    # 2019-3-29:
    # remove the padding, the last points will be discard

    #pad_length = int((nf-1)*inc+win_len)
    #zeros = np.zeros((pad_length - data_len, ))
    #pad_signal = np.concatenate((data, zeros))

    indices = np.tile(np.arange(0,win_len), (nf,1))+ np.tile(np.arange(0, nf*inc, inc), (win_len,1)).T 
    indices = np.array(indices, dtype=np.int32)
    frames = data[indices]
    windows = np.reshape(np.tile(window, nf), [nf, win_len])
    return frames*windows

def splice_feats(data, left=0, right=0):
    length, dims = data.shape
    sfeats = []
    # left 
    for i in range(left, 0, -1):
        t = data[:length-i]
        for j in range(i):
            t = np.pad(t, ((1, 0), (0, 0)), 'symmetric')
        sfeats.append(t)
    sfeats.append(data)
    # right
    for i in range(1,right+1):
        t = data[i:]
        for j in range(i):
            t = np.pad(t, ((0, 1 ), (0, 0)), 'symmetric')
        sfeats.append(t)
    return np.concatenate(np.array(sfeats), 1)


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    # if len(wave_data.shape) > 1:
    #     if wave_data.shape[1] == 1:
    #         wave_data = wave_data[0]
    #     else:
    #         wave_data = np.mean(wave_data, axis=-1)
    return wave_data

def channelnumread(path, fs=16000):
    wave_data, sr = sf.read(path)
    assert fs == sr
    return len(wave_data.shape)

def audiowrite(path, data, sample_rate=16000):
    # fs=sample_rate 
    # if len(data.shape) == 1:
    #     data = np.expand_dims(data, axis=1)
    sf.write(path, data, sample_rate)



def addnoise(clean_path, noise_path, rir_path, scale, snr):
    '''
    if rir is not None, the speech of noisy has reverberation
    and return the clean with reverberation
    else no reverberation
    Args:
        :@param clean_path: the path of a clean wav
        :@param noise_path: the path of a noise wav
        :@param start: the start point of the noise wav 
        :@param scale: the scale factor to control volume
        :@param snr:   the snr when add noise
    Return:
        :@param Y: noisy wav
        :@oaram X: clean wav
    '''
    clean = audioread(clean_path)
    noise = audioread(noise_path)
    
    noise_length = noise.shape[0]
    clean_length = clean.shape[0]
    clean_snr = snr
    noise_snr = -snr
    clean_weight = 10**(clean_snr/20)
    noise_weight = 10**(noise_snr/20)
    
    if clean_length > noise_length:
        start = np.random.randint(clean_length - noise_length)
        clean_selected = np.zeros(noise_length)
        clean_selected = clean[start : start + noise_length]
        noise_selected = noise
    elif clean_length < noise_length: 
        start = np.random.randint(noise_length - clean_length)
        noise_selected = np.zeros(clean_length)
        noise_selected = noise[start : start+clean_length]
        clean_selected = clean
    else:
        noise_selected = noise
        clean_selected = clean

    if rir_path is not None:
        rir = audioread(rir_path)
        rir1 = rir[:,0:2]
        rir2 = rir[:,2:4]
        rir_clean = add_reverb(clean_selected, rir1)
        if(channelnumread(noise_path) == 1):
            rir_noise = add_reverb(noise_selected, rir2)
        else:
            rir_noise = noise_selected
    else:
        rir_clean = clean_selected
        rir_noise = noise_selected

    '''
    # noise_n = activelev(noise_selected)
    # rir_clean_n = activelev(rir_clean)
    # rir_clean = rir_clean_n * clean_weight
    # noise = noise_n * noise_weight
    # noisy = rir_clean + noise
    # max_amp = np.max(np.abs([noise, rir_clean, noisy]))
    # mix_scale = 1/max_amp*scale
    # X = rir_clean * mix_scale
    # Y = noisy * mix_scale
    '''
    rir_noise_n_1 = activelev(rir_noise[:,0])
    rir_clean_n_1 = activelev(rir_clean[:,0])
    rir_noise_n_2 = activelev(rir_noise[:,1])
    rir_clean_n_2 = activelev(rir_clean[:,1])

    rir_clean[:,0] = rir_clean_n_1 * clean_weight
    rir_clean[:,1] = rir_clean_n_2 * clean_weight
    rir_noise[:,0] = rir_noise_n_1 * noise_weight
    rir_noise[:,1] = rir_noise_n_2 * noise_weight

    rir_clean = rir_clean[:len(rir_clean)-7999]
    if(channelnumread(noise_path) == 1):
        rir_noise = rir_noise[:len(rir_noise)-7999]

    noisy = rir_clean + rir_noise

    max_amp_0 = np.max(np.abs([rir_noise[:,0], rir_clean[:,0], noisy[:,0]]))
    max_amp_1 = np.max(np.abs([rir_noise[:,1], rir_clean[:,1], noisy[:,1]]))
    
    if max_amp_0 == 0:
        max_amp_0 = 1
    if max_amp_1 == 0:
        max_amp_1 = 1

    mix_scale_0 = 1/max_amp_0*scale
    mix_scale_1 = 1/max_amp_1*scale
    
    # if mix_scale_0 == 0:
    #     mix_scale_0 = 1
    # if mix_scale_1 ==0:
    #     mix_scale_1 = 1

    X = np.empty((rir_clean.shape))
    Y = np.empty((noisy.shape))
    X[:,0] = rir_clean[:,0] * mix_scale_0
    X[:,1] = rir_clean[:,1] * mix_scale_1
    Y[:,0] = noisy[:,0] * mix_scale_0
    Y[:,1] = noisy[:,1] * mix_scale_1
    return Y, X


def add_reverb(cln_wav, rir_wav):
    """
    Args:
        :@param cln_wav: the clean wav
        :@param rir_wav: the rir wav
    Return:
        :@param wav_tgt: the reverberant signal
    """
    
    rir_wav = np.array(rir_wav)
    wav_tgt_c1 = sp.convolve(cln_wav, rir_wav[:,0])
    wav_tgt_c2 = sp.convolve(cln_wav, rir_wav[:,1])
    # max_idx = np.argsort(rir_wav)[-1]
    # wav_tgt_c1 = wav_tgt_c1 / np.max(np.abs(wav_tgt_c1)) * np.max(np.abs(cln_wav))
    # wav_tgt_c2 = wav_tgt_c2 / np.max(np.abs(wav_tgt_c2)) * np.max(np.abs(cln_wav))

    wav_tgt = np.vstack((wav_tgt_c1, wav_tgt_c2))
    wav_tgt = np.transpose(wav_tgt)
    return wav_tgt
    # return wav_tgt[max_idx: max_idx + len(cln_wav)]


class Processer(object):
    
    def __init__(
                self, 
                win_len=400,
                win_inc=100,
                left_context=0,
                right_context=0,
                fft_len=512,
                snr_range=[0, 15],
                scale=0.9,
                window_type='hamming',
                log_path='./',
                running_stage='train'):

        self.left_context = left_context
        self.right_context = right_context
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.snr_range = snr_range
        self.scale=0.9
        self.window = {
                        'hamming': np.hamming(self.win_len)/1.2607934,
                        'none': np.ones(self.win_len)
                      }[window_type]
        self.rfft = np.fft.rfft
        self.log_path = log_path
        if running_stage=='train':
            self.log = open(self.log_path + '/clean_noise_rir_train.log','w')
        else:
            self.log = open(self.log_path + '/clean_noise_rir_validation.log','w')

    def process(self, clean_wav_path, noise_wav_path, rir_wave):
        
        if isinstance(self.snr_range, list):
            snr = (self.snr_range[-1] - self.snr_range[0])*np.random.ranf()+self.snr_range[0]
        else:
            snr = snr
        t = np.random.randn()*0.5+self.scale
        if t < 0:
            t = 1e-1
        elif t > 1:
            t = 1
        scale = t
        inputs, labels = addnoise(clean_wav_path, noise_wav_path, rir_wave, scale, snr)
        self.log.writelines(clean_wav_path+' '+noise_wav_path+' '+rir_wave+' '+str(snr)+'\n')
        self.log.flush()
        # inputs = enframe(inputs, self.window, self.win_len, self.win_inc)
        # inputs = np.abs(self.rfft(inputs, self.fft_len))
        # sinputs = splice_feats(
        #                         inputs,
        #                         left=self.left_context,
        #                         right=self.left_context
        #                     )

        # labels = enframe(labels, self.window, self.win_len, self.win_inc)
        # labels = np.abs(self.rfft(labels, self.fft_len))
        # slabels = splice_feats(
        #                         labels,
        #                         left=self.left_context,
        #                         right=self.left_context
        #                     )
        return inputs, labels

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = np.array([len(inputs), max_t, inputs[0].shape[1]])
    inputs_mat = np.zeros(shape, np.float32)
    for idx, inp in enumerate(inputs):
        inputs_mat[idx, :inp.shape[0], :] = inp
    return inputs_mat


def collate_fn(data):
    inputs, labels, lens = zip(*data)
    idx = sorted(enumerate(lens), key=lambda x :x[1], reverse=True)
    idx = [x[0] for x in idx]
    lens = [lens[x] for x in idx]
    padded_inputs = zero_pad_concat(inputs)
    padded_labels = zero_pad_concat(labels)
    return torch.from_numpy(padded_inputs[idx]), torch.from_numpy(padded_labels[idx]), torch.from_numpy(np.array(lens))


def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({'path': tmp[0], 'duration': float(tmp[1])})
            else:
                path_list.append({'path': tmp[0]})

class TFDataset(Dataset):
    def __init__(
                self,
                clean_scp,
                noise_scp,
                rir_scp,
                processer,
                use_chunk=False,
                repeat=5,
                chunk_size=4,
                SAMPLE_RATE=16000
            ):
        
        super(TFDataset, self).__init__()
        mgr = mp.Manager()
        self.processer = processer
        self.clean_wav_list = mgr.list()
        self.noise_wav_list = mgr.list()
        self.rir_wav_list = mgr.list()
        self.segement_length = chunk_size * SAMPLE_RATE
        self.index = mgr.list()
        pc_list = []
        # read wav config list
        
        p = mp.Process(target=parse_scp, args=(clean_scp, self.clean_wav_list))
        p.start()
        pc_list.append(p)        
        p = mp.Process(target=parse_scp, args=(noise_scp, self.noise_wav_list))
        p.start()
        pc_list.append(p)
        if rir_scp is not None:
            p = mp.Process(target=parse_scp, args=(rir_scp, self.rir_wav_list))
            p.start()
            pc_list.append(p)
        else:
            self.rir_wav_list = None

        for p in pc_list:
            p.join()
        # clip the wave to segement

        if use_chunk:            
            self._dochuck(SAMPLE_RATE=SAMPLE_RATE)
        else:
            self.index = [idx for idx in range(len(self.clean_wav_list))]

        # repeat index list for more input data
        self.index *= repeat
        self.size = len(self.index)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        item = self.index[index]
        noise = self.noise_wav_list[np.random.randint(len(self.noise_wav_list))]['path']
        if self.rir_wav_list is None:
            rir = None
        else:
            rir = self.rir_wav_list[np.random.randint(len(self.rir_wav_list))]['path']
        if isinstance(item,list):
            clean, start = item
            inputs, labels = self.processer.process(clean, noise, rir)
            length = inputs.shape[0]
            start = int(start)
            if start == -1:
                stack_length = self.segement_length - length
                stacked_inputs = inputs[:stack_length]
                stacked_labels = labels[:stack_length]
                inputs = np.concatenate([inputs, stacked_inputs], axis=0)
                labels = np.concatenate([labels, stacked_labels], axis=0)
            else:
                end_index = start + self.segement_length
                inputs = inputs[start : end_index]
                labels = labels[start : end_index]

        else:
            clean = item
            clean = self.clean_wav_list[clean]['path']
            inputs, labels = self.processer.process(clean, noise, rir_clean, rir_noise)
        inputs = np.stack((inputs[:,0]-inputs[:,1],
                           inputs[:,0],
                           inputs[:,1],),axis=0)
        labels = labels[:,0]
        # inputs = np.transpose(inputs)
        if inputs.shape[0] < 64000:
            inputs = np.pad(inputs,((0,0),(0,64000-inputs.shape[1])), 'constant', constant_values = (0,0))
            labels = np.pad(labels,(0,64000-labels.shape[0]), 'constant', constant_values = (0,0))
        elif inputs.shape[0] > 64000:
            inputs = inputs[:64000]
            labels = labels[:64000]
        return inputs, labels, inputs.shape[0]

    def _dochuck(self, SAMPLE_RATE=16000, num_threads=12):
        # mutliproccesing

        def worker(target_list, result_list, start, end, segement_length, SAMPLE_RATE):
            for item in target_list[start:end]:
                path = item['path']
                # duration = item['duration']
                length = audioread(path)
                length = length.shape[0]
                segement_length = self.segement_length
                # length = duration*SAMPLE_RATE
                # print("length")
                # print(length)
                if length < segement_length:
                    if length * 2 < segement_length:
                        continue
                    result_list.append([path, -1])
                else:
                    sample_index = 0
                    while sample_index + segement_length < length:
                        result_list.append(
                            [path, sample_index])
                        sample_index += segement_length
                    if sample_index != length - 1:
                        result_list.append([
                            path,
                            length - segement_length,
                        ])
        pc_list = []
        stride = len(self.clean_wav_list) // num_threads
        if stride < 100:
            p = mp.Process(
                            target=worker,
                            args=(
                                    self.clean_wav_list,
                                    self.index,
                                    0,
                                    len(self.clean_wav_list),
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                        )
            p.start()
            pc_list.append(p)
        else: 
            for idx in range(num_threads):
                if idx == num_threads-1:
                    end = len(self.clean_wav_list)
                else:
                    end = (idx+1)*stride
                p = mp.Process(
                                target=worker,
                                args=(
                                    self.clean_wav_list,
                                    self.index,
                                    idx*stride,
                                    end,
                                    self.segement_length,
                                    SAMPLE_RATE,
                                )
                            )
                p.start()
                pc_list.append(p)
        for p in pc_list:
            p.join()

class Sampler(tud.sampler.Sampler):
    '''
     
    '''
    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i+batch_size)
                        for i in range(0, it_end, batch_size)]
        self.data_source = data_source
        
    def __iter__(self):
        np.random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)
    
def make_loader(clean_scp, noise_scp, rir_scp=None, batch_size=8, use_chunk=False, repeat=1, SAMPLE_RATE=16000, chunk_size=4, num_threads=12, processer=Processer()):
    '''
        clean_scp, clea
    '''
    dataset = TFDataset(
                        clean_scp,
                        noise_scp,
                        rir_scp,
                        processer,
                        use_chunk=use_chunk,
                        chunk_size=chunk_size,
                        repeat=repeat,
                        SAMPLE_RATE=SAMPLE_RATE
                    )

    sampler = Sampler(dataset, batch_size)

    if use_chunk:
        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_threads,
                sampler=sampler,
                drop_last=False
            )

    else:
        loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_threads,
                sampler=sampler,
                collate_fn=collate_fn,
                drop_last=False
            )
    return loader

if __name__ == '__main__':
    processer = Processer()
    # print(processer)
#    print('Test TFdataset')
#    print('using chunk')
#    dataset = TFDataset('./clean.lst', 'noise.lst', None, processer, use_chunk=True, chunk_size=1, repeat=2)
#    print(len(dataset))
#    stime = time.time()
    #for (idx, item) in enumerate(dataset):
    #    if (idx+1) % 100 == 0:
    #        inputs, labels, length = item
    #        etime = time.time()
    #        print(idx, etime - stime, length)
    #        stime = etime
#    print('-----------------------------------------')
#    print('using sentence')
    # dataset = TFDataset('../../speech_noise_rir_list/1kh_data.txt', '../../speech_noise_rir_list/noise_singlechannel+record.txt', '../../speech_noise_rir_list/rir.txt', processer, use_chunk=True, repeat=1)
    # print(len(dataset))
    # print(dataset)
    dataset = TFDataset('clean_test.lst', 'noise_test.lst', 'rir_test.lst', processer, use_chunk=True, repeat=1)
    for (idx, item) in enumerate(dataset):
        inputs, labels, length = item
        print(inputs.shape)
        print(labels.shape)
        audiowrite("./input/"+str(idx)+".wav", inputs, sample_rate=16000)
        audiowrite("./label/"+str(idx)+".wav", labels, sample_rate=16000)
        # print(inputs.shape)
        # print(labels.shape)
    #     sio.savemat(str(idx)+'.mat', {'inputs':inputs,'labels':labels})
#        if (idx+1) % 100 == 0:
#            inputs, labels, length = item
#            etime = time.time()
#            print(idx, etime - stime, length)
#            stime = etime
#    print('-----------------------------------------')
#    print('Test Dataloader')
#    loader = make_loader('./clean.lst', 'noise.lst',None, use_chunk=False, repeat=10, num_threads=18)
#    for (idx, item) in enumerate(loader):
#        if (idx+1) % 100 == 0:
#            inputs, labels, length = item
#            etime = time.time()
#            print(idx, (etime - stime)/100., length)
#            stime = etime


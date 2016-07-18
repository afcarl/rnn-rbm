from __future__ import print_function

import os

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.signal import spectrogram


def linear_smooth(x, window_length):
    s = np.r_[x[window_length-1:0:-1],x,x[-1:-window_length:-1]]
    w = np.ones(window_length,'d')
    return np.convolve(w/w.sum(), s, mode='valid')


class DirectoryIterator(object):
    def __init__(self, dir_path):
        assert os.path.isdir(dir_path), 'Invalid directory: "%s"' % dir_path
        self.dir_path = dir_path

    def __iter__(self):
        for file in os.listdir(self.dir_path):
            if file.lower().endswith('.wav'):
                try:
                    rate, data = wavread(os.path.join(self.dir_path, file))
                    print('%s (rate: %d)' % (file, rate))
                    yield data, rate
                except Exception, e:
                    print('Error reading "%s" (%s)' % (file, e))
            else:
                print('Unsupported filetype: "%s"' % file)


def parse_segments(data, rate, threshold=90, threshold_value=None, buffer=5, min_spacing=2,
                   min_length=15, max_length=500, window_length=4):
    # data = data[:10000000]

    # convert constants from ms to frames
    convert = lambda x: x * rate / float(1000)
    window_length = convert(window_length)
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    min_spacing = convert(min_spacing)
    min_length = convert(min_length)
    max_length = convert(max_length)
    buffer = int(convert(buffer))

    # rectify and smooth the data
    rectified = np.abs(data)
    smoothed = linear_smooth(rectified, window_length)
    smoothed = np.abs(smoothed)

    # calculate threshold change points
    if threshold_value is None:
        threshold_value = np.percentile(smoothed, threshold, interpolation='linear')
    indices = smoothed >= threshold_value
    bounded = np.hstack(([0], indices, [0]))
    diffs = np.diff(bounded)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]

    # join segments
    for i in range(len(run_starts)-1):
        if run_starts[i+1] - run_ends[i] < min_spacing:
            run_starts[i+1] = -1
            run_ends[i] = -1
    run_starts = [i for i in run_starts if i != -1]
    run_ends = [i for i in run_ends if i != -1]

    # remove segments that are too small
    run_starts, run_ends = map(list, zip(*[(s, e) for s, e in zip(run_starts, run_ends)
                                           if min_length < e - s < max_length]))
    for start, end in zip(run_starts, run_ends):
        f_spec, t_spec, data_spec = spectrogram(data[start-buffer:end+buffer], rate, noverlap=128, nperseg=256)
        yield f_spec, t_spec, data_spec

if __name__ == '__main__':
    assert 'VOCALIZATION_FILES' in os.environ, 'Must set "VOCALIZATION_FILES" environment variable'
    files = DirectoryIterator(os.environ['VOCALIZATION_FILES'])
    save_dir = os.path.join(os.environ['VOCALIZATION_FILES'], 'spectrograms')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)
    i = 0
    info = dict()
    for data, rate in files:
        for f, t, data_spec in parse_segments(data, rate):
            print('\r%d ' % i, end='')
            if len(data_spec) == 0:
                continue
            np.save(open('spectrogram_%d.npz' % i, 'wb'), data_spec)
            i += 1
            info['FREQ_DIM'] = len(f)
            info['FREQ_AXIS_MIN'] = f[0]
            info['FREQ_AXIS_MAX'] = f[len(f)-1]
    import pickle as pkl
    info['NUM_FILES'] = i
    pkl.dump(info, open('info.pkl', 'wb'))

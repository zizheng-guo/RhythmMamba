"""The dataloader for VIPL-HR dataset."""
import glob
import os
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
import csv

class VIPLHRLoader(BaseLoader):
    """The data loader for the VIPL-HR dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an VIPL dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |p1|
                     |  |v1|
                     |      |source1|
                     |          |video.avi|
                     |          |time.txt|
                     |          |wave.csv|
                     |      |source2/
                     |          |video.avi|
                     |          |wave.csv|
                     |p1|
                     |  |v2|
                     |      |source1|
                     |          |video.avi|
                     |          |time.txt|
                     |          |wave.csv|
                     |      |source2/
                     |          |video.avi|
                     |          |wave.csv|
                     |...
                     |
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)


    def get_raw_data(self, data_path):
        """Returns data directories under the path (For VIPL-HR dataset)."""
        data_dirs = sorted(glob.glob(data_path + os.sep + "p*"))
        data_vs=['v1','v1-2','v2','v3','v4','v5','v6','v7','v8','v9']
        data_sources=['source1','source2','source3']
        if not data_dirs:
            raise ValueError("Data paths empty!")

        dirs = []
        for data_dir in data_dirs:
            for data_v in data_vs:
                v_path = os.path.join(data_dir, data_v)
                if os.path.exists(v_path):
                    for data_source in data_sources:
                        source_path = os.path.join(v_path, data_source)
                        if os.path.exists(source_path):
                            subject_num = os.path.split(data_dir)[-1].replace('p', '')
                            index = f"p{subject_num}_v{data_v[1:]}_s{data_source[6:]}"
                            subject = int(subject_num[0:2])
                            dirs.append({"index": index, "path": source_path, "subject": subject})
        return dirs
    
    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs
            begin: train / test
            end: fold
        """
        fold = []
        if end == 1:
            fold = [ 74, 100,  21,  10,  65,  90,  57,  58, 101,  59,  64,   7,  49, 43,  62,  25,  50, 105,  35,  81,  12]
        elif end == 2:
            fold = [ 88,   9,  24,  55,  38,  22,   6,  15,  34,  29,   2,  17,  31, 51,  23,  27,  40,  20,  13,  79, 103]
        elif end == 3:
            fold = [  1,  63,  61,  82,  46,   5,  78,  56, 106,  67,  84,  99,  97, 70,  87,   8,  48,  76,  91, 107,  92]
        elif end == 4:
            fold = [ 89, 39,  3, 60, 18, 14, 69, 19, 93, 16, 86,  4, 52, 11, 33, 42, 96, 30, 54, 36, 80, 32]
        elif end == 5:
            fold = [ 71,  26,  53,  41, 104,  85,  66,  44, 102,  75,  72,  47,  28, 94,  83,  98,  68,  37,  95,  73,  45,  77]

        data_dirs_new = []

        if begin == 0: #test
            for data in data_dirs:
                subject = data['subject']
                data_dir = data['path']
                index = data['index']
                if subject in fold: 
                    data_dirs_new.append({"index": index, "path": data_dir, "subject": subject})
        else: #train
            for data in data_dirs:
                subject = data['subject']
                data_dir = data['path']
                index = data['index']
                if subject not in fold: 
                    data_dirs_new.append({"index": index, "path": data_dir, "subject": subject})            
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        "invoked by preprocess_dataset for multi_process."
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(os.path.join(data_dirs[i]['path'],"video.avi"))
        bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "wave.csv"))
        time = self.read_fps(os.path.join(data_dirs[i]['path'], "time.txt"))

        if time != 0:
            frames = BaseLoader.resample_video(frames, int(time * 30 / 1000))

        bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        "Reads a video file, returns frames(T, H, W, 3) "
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = VidObj.read()
        VidObj.release()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        bvp = []
        with open(bvp_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                bvp.append(float(row[0]))
        return np.asarray(bvp)
    
    @staticmethod
    def read_fps(fps_file):
        if not os.path.exists(fps_file):  
            return 0
        value = 0
        with open(fps_file, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            last_line = last_line.strip()  
            value = float(last_line)  
        return value
    
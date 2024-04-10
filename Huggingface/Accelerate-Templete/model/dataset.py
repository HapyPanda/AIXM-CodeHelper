import os.path
import math
import pickle
import warnings

import glob

from torch.utils.data import Dataset
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips


# Process Video Data
def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video # C T H W , H=W=resolution 


# Dataset Model
class VideoDataset(Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length,cache_folder=None,train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()

        # Initial Setting
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.data_folder = data_folder
        self.cache_folder = cache_folder

        # Glob .avi files 
        files = sum([glob.glob(os.path.join(self.data_folder, '**', f'*.{ext}'), recursive=True) for ext in self.exts], []) # sum operation is for multi-video-exts,combine them in one list

        # # hacky way to compute # of classes (count # of unique parent directories)
        # self.classes = list(set([get_parent_dir(f) for f in files]))
        # self.classes.sort()
        # self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        # Cache data for fast reload
        warnings.filterwarnings('ignore')
        cache_file = os.path.join(self.data_folder, f"metadata_{sequence_length}.pkl") if cache_folder==None else os.path.join(self.cache_folder, f"metadata_{sequence_length}.pkl")
        if not os.path.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=4)
            with open(cache_file,'wb') as f:
                pickle.dump(clips.metadata, f)
        else:
            with open(cache_file,'rb') as f:
                metadata = pickle.load(f)
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips

    # @property
    # def n_classes(self):
    #     return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, audio, info, idx = self._clips.get_clip(idx)
        video_name = self._clips.video_paths[idx]
        # label = self.class_to_label[class_name]
        return dict(video=preprocess(video, resolution),name=video_name)

        
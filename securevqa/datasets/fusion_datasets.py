import decord
from decord import VideoReader
import glob
import os.path as osp
import torch, torchvision
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50
from torchvision import transforms
from PIL import Image
import numpy as np
import random
random.seed(20240117)

decord.bridge.set_bridge("torch")

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])

        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            fast_feature = nn.functional.adaptive_avg_pool3d(x[1], output_size=(1, 1, 1))    #Only the embeddings of fast path are utilized

        return fast_feature.squeeze()



def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list

def get_resize_samples(         ###  Resize V^{inter}
        video_path,
        resize_size
):
    vreader = VideoReader(video_path)

    motion_frame_dict = {idx: vreader[idx] for idx in np.arange(len(vreader))}

    motion_imgs = [motion_frame_dict[idx] for idx in range(len(vreader))]     # V^{inter}
    motion_video = torch.stack(motion_imgs, 0).permute(3, 0, 1, 2)

    transform = transforms.Compose([transforms.Resize([resize_size, resize_size])])
    motion_video = np.array(motion_video.permute(1, 2, 3, 0))
    video_length = motion_video.shape[0]
    video_channel = motion_video.shape[3]

    transformed_video = np.zeros([video_length, resize_size, resize_size, video_channel])
    for frame_idx in range(video_length):
        frame = motion_video[frame_idx]
        frame = Image.fromarray(np.uint8(frame))
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    motion_target_video = torch.from_numpy(transformed_video).permute(3, 0, 1, 2)    # F^{inter}

    return motion_target_video



class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling

        super().__init__()

        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        self.phase = opt["phase"]
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.slowfast = slowfast().to('cuda')
        self.resize_size = opt["Resize_size"]
        self.d_num = opt["d_num"]      # d in Equation 2
        self.segments_num = opt["segments_num"]  # splits the video into 16 segments in the inter-frame branch

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            try:
                with open(self.ann_file, "r") as fin:
                    for line in fin:
                        line_split = line.strip().split(",")
                        filename, _, _, label = line_split
                        label = float(label)
                        filename = osp.join(self.data_prefix, filename)
                        self.video_infos.append(dict(filename=filename, label=label))
            except:
                #### No Label Testing
                video_filenames = sorted(glob.glob(self.data_prefix + "/*.mp4"))
                # print(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))


    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]

        Inter_data = get_resize_samples(filename,self.resize_size)  #Get F^{inter}

        guardian = (2 * np.random.randint(0, 2, size=(Inter_data.shape[0] * (Inter_data.shape[2]) * (Inter_data.shape[3]))) - 1)  # Initialized guardian map from {-1,1}
        guardian = torch.from_numpy(guardian.reshape(Inter_data.shape[0], 1, (Inter_data.shape[2]), (Inter_data.shape[3])))

        Inter_data = torch.clamp((Inter_data + guardian), 0, 255)
        Inter_data = ((Inter_data.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        video_name = osp.basename(video_info["filename"])

        video_clip = self.segments_num
        video_length = Inter_data.shape[1]
        video_frame_rate = int(video_length / 16)
        d_num = self.d_num                # Number of frames per segment
        transformed_video_all = []
        for i in range(video_clip):   # Divide the video into 16 segments
            transformed_video = torch.zeros([3, d_num, self.resize_size, self.resize_size])
            if (i * video_frame_rate + d_num) <= video_length:
                transformed_video = Inter_data[:, i * video_frame_rate: (i * video_frame_rate + d_num)]
            else:
                transformed_video[:, :(video_length - i * video_frame_rate)] = Inter_data[:, i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), d_num):
                    transformed_video[:, j] = transformed_video[:, video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)


        Inter_frame_information = torch.zeros([video_clip, 256])           # E^{inter}, and the dimension of the inter branch is 256
        for idx, ele in enumerate(transformed_video_all):
            ele = ele.unsqueeze(0).to('cpu').float()
            inputs = pack_pathway_output(ele, 'cpu')
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda()
            fast_feature = self.slowfast(inputs)             # Only the embeddings in fast path are utilized
            Inter_frame_information[idx] = fast_feature

        return Inter_frame_information, video_name

    def __len__(self):
        return len(self.video_infos)




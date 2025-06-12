import decord
from decord import VideoReader
import glob
import os.path as osp
import torch, torchvision
from functools import lru_cache
import random
import copy
import skvideo.io
from torchvision import transforms
random.seed(42)

decord.bridge.set_bridge("torch")

def get_spatial_fragments_and_motion_resize(
        video,
        motion_video,
        reize_size,
        fragments_h=7,
        fragments_w=7,
        fsize_h=32,
        fsize_w=32,
        aligned=32,
        nfrags=1,
        random=False,
        random_upsample=False,
        fallback_type="upsample",
        **kwargs,
):
    transform = transforms.Compose([transforms.Resize([reize_size, reize_size])])    # Resize V^{inter}
    video_length = motion_video.shape[1]
    video_channel = motion_video.shape[0]
    transformed_video = torch.zeros([video_channel,video_length, reize_size, reize_size])
    for frame_idx in range(video_length):
        frame = motion_video[:,frame_idx,:,:] * 255
        frame = frame.type(torch.uint8)
        frame = transform(frame)
        transformed_video[:,frame_idx,:,:] = frame

    motion_target_video = transformed_video / 255

    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w

    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:
        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )

    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)


    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w

                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                             :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                             ]

    return target_video, motion_target_video


import numpy as np
import random


class FragmentSampleFrames:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0., ):

        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.size_t = fragments_t * fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames, train=False):

        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t

        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)

        ranges_t = (
                np.arange(self.fsize_t)[None, :] * self.frame_interval
                + rnd_t[:, None]
                + tgrids[:, None]
        )

        drop = random.sample(list(range(self.fragments_t)), int(self.fragments_t * self.drop_rate))
        dropped_ranges_t = []
        for i, rt in enumerate(ranges_t):
            if i not in drop:
                dropped_ranges_t.append(rt)
        return np.concatenate(dropped_ranges_t)

    def __call__(self, total_frames, train=False, start_index=0):
        frame_inds = []

        for i in range(self.num_clips):
            frame_inds += [self.get_frame_indices(total_frames)]

        frame_inds = np.concatenate(frame_inds)
        frame_inds = np.mod(frame_inds + start_index, total_frames)
        return frame_inds.astype(np.int32)



class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        ## opt is a dictionary that includes options for video sampling

        super().__init__()
        # self.key = key
        self.video_infos = []
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.opt = opt
        self.sample_types = opt["sample_types"]
        self.data_backend = opt.get("data_backend", "disk")
        self.augment = opt.get("augment", False)

        self.phase = opt["phase"]
        self.crop = opt.get("random_crop", False)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        self.samplers = {}

        for stype, sopt in opt["sample_types"].items():
            if "t_frag" not in sopt:
                # revised legacy temporal sampling
                self.samplers[stype] = FragmentSampleFrames(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
            else:
                self.samplers[stype] = FragmentSampleFrames(sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                                                            sopt["frame_interval"], sopt["num_clips"])

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
                print(video_filenames)
                for filename in video_filenames:
                    self.video_infos.append(dict(filename=filename, label=-1))

    def refresh_hypers(self):
        if not hasattr(self, "initial_sample_types"):
            self.initial_sample_types = copy.deepcopy(self.sample_types)

        types = self.sample_types

        if "fragments_up" in types:
            ubh, ubw = self.initial_sample_types["fragments_up"]["fragments_h"] + 1, \
                       self.initial_sample_types["fragments_up"]["fragments_w"] + 1
            lbh, lbw = self.initial_sample_types["fragments"]["fragments_h"] + 1, \
                       self.initial_sample_types["fragments"]["fragments_w"] + 1
            dh, dw = types["fragments_up"]["fragments_h"], types["fragments_up"]["fragments_w"]

            types["fragments_up"]["fragments_h"] = random.randrange(max(lbh, dh - 1), min(ubh, dh + 2))
            types["fragments_up"]["fragments_w"] = random.randrange(max(lbw, dw - 1), min(ubw, dw + 2))

        if "resize_up" in types:
            types["resize_up"]["size_h"] = types["fragments_up"]["fragments_h"] * types["fragments_up"]["fsize_h"]
            types["resize_up"]["size_w"] = types["fragments_up"]["fragments_w"] * types["fragments_up"]["fsize_w"]

        self.sample_types.update(types)


    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]

        label = video_info["label"]
        return filename, label

    def __len__(self):
        return len(self.video_infos)




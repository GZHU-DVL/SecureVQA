import torch
import torchvision.transforms as transforms
import scipy.stats
import numpy as np
import os
import argparse
import yaml
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
import securevqa.models_attack as models
import securevqa.datasets_attack as datasets
from securevqa.datasets_attack.fusion_datasets import FragmentSampleFrames,get_spatial_fragments_and_motion_resize
import wandb
from tqdm import tqdm
from decord import VideoReader
from pytorchvideo.models.hub import slowfast_r50
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_num_threads(3)
seed = 202401017
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = True

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

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
            fast_feature = nn.functional.adaptive_avg_pool3d(x[1], output_size=(1, 1, 1))
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

def l2_proj(adv, orig, eps=15.0):   # Limit the pixel-level L2 norm of perturbations within 1/255
    delta = adv - orig
    norm_bs = torch.norm(delta.contiguous().view(delta.shape[0], -1), dim=1)
    out_of_bounds_mask = (norm_bs > eps).float()
    x = (orig + eps*delta/norm_bs.view(-1,1,1,1))*out_of_bounds_mask.view(delta.shape[0], 1, 1, 1)
    x += adv*(1-out_of_bounds_mask.view(-1, 1, 1, 1))
    return x

def l2_proj_linf(adv, orig, eps=15.0):  # Limit the Linf norm of perturbations within 3/255
    delta = adv - orig
    delta = torch.clamp(delta,-3/255,3/255)
    x = orig + delta
    return x

sample_types = ["fragments"]
def jnd_attack_adam(inds_list,video_data,Slowfast, model, label, median, q_hat_original_min, q_hat_original_max, opt, config):  #White-box attack on NR-VQA model
    adv = video_data.clone().detach()  #Adversarial video
    ref = video_data.clone().detach()  # Clean_video
    s_init = get_score(inds_list,ref, Slowfast,model, opt, 0, config) # Original quality score (estimated quality score)
    eps = (ref.shape[1] * ref.shape[2] * 3 * (1 / 255) ** 2) ** 0.5    #JND constraint
    adv.requires_grad = True
    if label >= median:   #Compute the boundary (disturbed quality score)
        boundary = torch.from_numpy(np.array(q_hat_original_min)).to('cuda')
    else:
        boundary = torch.from_numpy(np.array((q_hat_original_max))).to('cuda')
    for k in range(config.iterations):  #One round contains K iterations.
        s = get_score(inds_list, adv, Slowfast, model, opt, 0, config)
        if k == 0:
            init_noise = torch.randint(-1, 1, adv.size()).to(adv)   #Initialize the perturbations
            init_noise = init_noise.float() / 255
            adv = torch.clamp(adv.detach() + init_noise, 0, 1)
            adv.requires_grad = True
            optimizer = torch.optim.Adam([adv], lr=config.beta)
            s = get_score(inds_list,adv,Slowfast, model, opt, 0, config)

        Lsrb = F.l1_loss(s, boundary)    #Compute the Score-Reversed Boundary loss
        optimizer.zero_grad()
        Lsrb.backward()
        adv.grad.data[torch.isnan(adv.grad.data)] = 0
        adv.grad.data = adv.grad.data / ((adv.grad.data.reshape(adv.grad.data.size(0), -1) + 1e-12).norm(dim=1).view(-1,1,1,1))  #Update the adversarial video.
        optimizer.step()
        adv.data = l2_proj(adv.data, ref,eps)  # Limit the pixel-level L2 norm of perturbations within 1/255
        adv.data.clamp_(min=0, max=1)
    print("Lsrb", Lsrb)

    return adv, adv-ref, boundary  # Return adversarial video, perturbations, and boundary

def get_score(inds_list,video_data,Slowfast, model, opt, flag, config):
    '''
    flag = 1: Get the score of the original video or the adversarial video
    flag = 0: Get the score of a frame during the attack
    '''
    result = dict()
    video = {}
    all_frame_inds = []
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    sampler = FragmentSampleFrames(config.d_num, config.num_clips, config. num_clips)

    if flag == 1:
        Intra_data = {}
        Inter_data = {}
        frame_inds = sampler(len(video_data), False)
        all_frame_inds.append(frame_inds)
        all_frame_inds = np.concatenate(all_frame_inds, 0)
        inds_list.append(np.unique(all_frame_inds))
        # log("all_frame_inds: {}".format(all_frame_inds))
        frame_dict = {idx: video_data[idx] for idx in np.unique(all_frame_inds)}
        imgs = [frame_dict[idx] for idx in frame_inds]
        intra_video = torch.stack(imgs, 0).permute(3, 0, 1, 2)
        '''
        Since we only attack the frames selected by SecureVQA, 
        the selected frames are also used to get the inter-frame information during attack instead of all frames described in the paper. 
        We verified that this has minimal impact on the effectiveness of the attack.
        '''
        Intra_data['fragments'], Inter_data['resize'] = get_spatial_fragments_and_motion_resize(intra_video, intra_video, config.resize_size)  # Get F^{intra} and F^{inter}
        guardian = (2 * np.random.randint(0, 2, size=(Intra_data['fragments'].shape[0] * (Intra_data['fragments'].shape[2]) *
                                                  (Intra_data['fragments'].shape[3]))) - 1) * 1 / 255  # Initialized guardian map from {-1,1}
        guardian = torch.from_numpy(
            guardian.reshape(Intra_data['fragments'].shape[0], 1, (Intra_data['fragments'].shape[2]), (Intra_data['fragments'].shape[3])))

        for k, v in Intra_data.items():
            Intra_data[k] = torch.clamp((Intra_data[k] + guardian.to(Intra_data[k].device)), 0, 1)
            Intra_data[k] = ((v.permute(1, 2, 3, 0) - mean.to(v.device)) / std.to(v.device)).permute(3, 0, 1, 2)
        for k, v in Inter_data.items():
            Inter_data[k] = torch.clamp((Inter_data[k].to(Inter_data[k].device) + guardian), 0, 1)
            Inter_data[k] = ((v.permute(1, 2, 3, 0) - mean.to(v.device)) / std.to(v.device)).permute(3, 0, 1, 2)

        segments_num = config.segments_num      #

        video_length = Inter_data['resize'].shape[1]
        video_frame_rate = int(video_length / segments_num)
        d_num = config.d_num
        transformed_video_all = []

        for i in range(segments_num):    # Divide the video into 16 segments
            transformed_video = torch.zeros([3, d_num, config.resize_size, config.resize_size])
            if (i * video_frame_rate + d_num) <= video_length:
                transformed_video = Inter_data['resize'][:,
                                    i * video_frame_rate: (i * video_frame_rate + d_num)]
            else:
                transformed_video[:, :(video_length - i * video_frame_rate)] = Inter_data['resize'][:,
                                                                               i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), d_num):
                    transformed_video[:, j] = transformed_video[:, video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        Inter_embeddings = torch.zeros([1,segments_num, 256])     # E^{inter}, and the dimension of the inter branch is 256
        for idx, ele in enumerate(transformed_video_all):
            ele = ele.unsqueeze(0).to('cpu').float()
            inputs = pack_pathway_output(ele, 'cpu')
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda()
            fast_feature = Slowfast(inputs)
            Inter_embeddings[0][idx] = fast_feature
        Inter_embeddings = Inter_embeddings.unsqueeze(0).repeat(4, 1, 1, 1)
        # It needs to be evaluated four clips during testing, the inter_embeddings are replicated four times to keep the dimensions consistent.

    else:
        Intra_data = {}
        Inter_data = {}
        fragments_video = torch.zeros((3, config.d_num * config.num_clips, config.fragments_size, config.fragments_size))
        fragments_motion_video = torch.zeros((3, config.d_num * config.num_clips, config.resize_size, config.resize_size))
        '''
        Four clips are needed to get the video quality, and each clip has 32 frames.
        Since the attack process optimizes one frame at a time, in order to speed up, the attacked frame is processed and then copied 32 times as a clip. 
        '''
        for i in range(0,config.d_num * config.num_clips, config.d_num):
            fragments,motion = get_spatial_fragments_and_motion_resize(video_data.permute(3, 0, 1, 2),video_data.permute(3, 0, 1, 2), config.resize_size)
            fragments_video[:,i:i+config.d_num,:,:] = fragments
            fragments_motion_video[:, i:i + config.d_num, :, :] = motion

        Intra_data['fragments'] = fragments_video
        Inter_data['resize'] = fragments_motion_video

        guardian = (2 * np.random.randint(0, 2, size=(Intra_data['fragments'].shape[0] * (Intra_data['fragments'].shape[2]) * (Intra_data['fragments'].shape[3]))) - 1) * 1 / 255
        guardian = torch.from_numpy(guardian.reshape(Intra_data['fragments'].shape[0], 1, (Intra_data['fragments'].shape[2]), (Intra_data['fragments'].shape[3])))

        for k, v in Intra_data.items():
            Intra_data[k] = torch.clamp((Intra_data[k] + guardian.to(Intra_data[k].device)), 0, 1)
            Intra_data[k] = ((v.permute(1, 2, 3, 0) - mean.to(v.device)) / std.to(v.device)).permute(3, 0, 1, 2)
        for k, v in Inter_data.items():
            Inter_data[k] = torch.clamp((Inter_data[k].to(Inter_data[k].device) + guardian), 0, 1)
            Inter_data[k] = ((v.permute(1, 2, 3, 0) - mean.to(v.device)) / std.to(v.device)).permute(3, 0, 1, 2)
        Inter_embeddings = torch.zeros([1, config.segments_num, 256])  # E^{inter}, and the dimension of the inter branch is 256

        inputs = pack_pathway_output(fragments_motion_video[:,:config.d_num,:,:].unsqueeze(0).to('cpu').float(), 'cpu')
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
        fast_feature = Slowfast(inputs)
        Inter_embeddings[0][:] = fast_feature
        Inter_embeddings = Inter_embeddings.unsqueeze(0).repeat(4, 1, 1, 1)
        # It needs to be evaluated four clips during testing, the inter_embeddings are replicated four times to keep the dimensions consistent.
    Intra_data["num_clips"] = config.num_clips

    for key in sample_types:
        if key in Intra_data:
            # print(data[key].shape)
            video[key] = Intra_data[key].unsqueeze(0).to('cuda')
            b, c, t, h, w = video[key].shape
            video[key] = video[key].reshape(b, c, Intra_data["num_clips"], t // Intra_data["num_clips"], h, w).permute(0, 2, 1, 3,
                                                                                                           4,
                                                                                                           5).reshape(
                b * Intra_data["num_clips"], c, t // Intra_data["num_clips"], h, w)

    result["pr_labels"] = torch.mean(model(video,Inter_embeddings))
    return result["pr_labels"]

def do_attack(inf_loader, model, device, opt, max_score,config):
    Slowfast = slowfast().to(device)
    q_mos = []
    q_hat_original = []
    q_hat_adv = []
    l2 = []
    R_value_list = []
    index_list = []
    videoname = []
    inds_list = []
    for i, (filename, label) in enumerate(tqdm(inf_loader, desc="Training")):
        q_mos.append(label[0] / max_score)
        video_data = VideoReader(filename[0])
        with torch.no_grad():
            sa = get_score(inds_list, video_data[0:len(video_data)] / 255, Slowfast, model, opt, 1, config)  # Original quality score (estimated quality score)
        q_hat_original.append(sa.item())
        videoname.append(filename[0])

    median = np.median(np.array(q_mos))        # The threshold to decide whether a video is of high quality or low quality
    q_hat_original_min = (((0)-np.mean(q_mos))/np.std(q_mos))*np.std(q_hat_original) + np.mean(q_hat_original) # Compute the boundary according to the distribution quality scores estimated by target NR-VQA models.
    q_hat_original_max = (((1)-np.mean(q_mos))/np.std(q_mos))*np.std(q_hat_original) + np.mean(q_hat_original)

    srcc_original = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat_original)[0]
    log("srcc_original = {}".format(srcc_original))

    krcc_original = scipy.stats.kendalltau(x=q_mos, y=q_hat_original)[0]
    log("krcc_original = {}".format(krcc_original))

    plcc_original = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat_original)[0]
    log("plcc_original = {}".format(plcc_original))

    rmse_original = np.sqrt(((np.array(q_mos) - np.array(q_hat_original)) ** 2).mean())
    log("rmse_original = {}".format(rmse_original))
    '''
    Because the attack process optimization efficiency is too low, 
    in the test process, we assume that the attacker knows which frames SecureVQA selects for evaluation, 
    and only attacks the selected frames. This has no effect on the effectiveness of the attack.
    '''
    '''
    If you want to further improve the efficiency, the FasterVQA strategy can be adopted to select only one clip to evaluate the quality.
     This can quickly verify the safety of the model. 
    '''
    for video_index in range(len(q_mos)):

        frame_index = inds_list[video_index]        # Frames selected by SecureVQA for evaluation
        start = time.time()
        video_data = VideoReader(videoname[video_index])
        perturbations = torch.zeros((len(video_data), video_data[0].shape[0],
                                               video_data[0].shape[1], video_data[0].shape[2]))
        adversarial_video = torch.zeros((len(video_data), video_data[0].shape[0],
                                 video_data[0].shape[1], video_data[0].shape[2]))


        if config.attack_trigger == 1:
            for index in range(0, len(frame_index)):
                video_data_round = video_data[frame_index[index]:frame_index[index] + 1] / 255
                video_data_round = video_data_round.to(device)
                adv_round, perturbations_round, boundary = jnd_attack_adam(inds_list, video_data_round, Slowfast, model, q_mos[video_index],
                                                                                                  median,
                                                                                                  q_hat_original_min,
                                                                                                  q_hat_original_max,
                                                                                                  opt, config
                                                                                                  )
                adversarial_video[frame_index[index]:frame_index[index] + 1] = adv_round.cpu().detach()  #Adversarial frames in one round
                perturbations[frame_index[index]:frame_index[index] + 1] = perturbations_round.cpu().detach()
            with torch.no_grad():
                sa_adv = get_score(inds_list, adversarial_video, Slowfast, model, opt, 1, config)    #Compute the estimated quality score of adversarial video
                R_value_list.append(np.log((abs((q_hat_original[video_index] - boundary).cpu())) / (abs((sa_adv - q_hat_original[video_index]).cpu()) + 1e-12))) #Compute the R_value
                l2.append((((torch.mean(torch.norm((perturbations).contiguous().view((perturbations).shape[0], -1), dim=1),dim=0))**2)
                           /(adversarial_video.shape[1]*adversarial_video.shape[2]*adversarial_video.shape[3]))**0.5)  #Compute the pixel-level L2 norm
                q_hat_adv.append(sa_adv.item())

        method_folder = 'SecureVQA'
        attack_folder = os.path.join(config.attack_folder, method_folder,file_name)
        print("attack_folder", attack_folder)
        if not os.path.exists(attack_folder):
            os.makedirs(attack_folder)

        original_folder = os.path.join(config.original_folder, method_folder,file_name)
        print("original_folder", original_folder)
        if not os.path.exists(original_folder):
            os.makedirs(original_folder)


        end = time.time()
        torch.cuda.empty_cache()
        log('{} seconds'.format(end - start))

    l2 = torch.mean(torch.Tensor(l2))
    log("l2 = {}".format(l2))

    f = open(str(attack_folder + 'score'), 'w', encoding='utf-8', newline='' "")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["MOS", "Original_scores", "Fake_scores"])
    for i in range(len(q_hat_original)):
        result = [q_mos[i].item(), q_hat_original[i], q_hat_adv[i]]
        csv_writer.writerow(result)

    R_value = torch.mean(torch.Tensor(R_value_list))
    log("R_value = {}".format(R_value))

    srcc_adv = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat_adv)[0]
    log("srcc_adv = {}".format(srcc_adv))

    krcc_adv = scipy.stats.kendalltau(x=q_mos, y=q_hat_adv)[0]
    log("krcc_adv = {}".format(krcc_adv))

    plcc_adv = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat_adv)[0]
    log("plcc_adv = {}".format(plcc_adv))

    rmse_adv = np.sqrt(((np.array(q_mos) - np.array(q_hat_adv)) ** 2).mean())
    log("rmse_adv = {}".format(rmse_adv))

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--attack_folder", type=str, default="./counterexample")
    parser.add_argument("--original_folder", type=str, default="./original")
    parser.add_argument("--attack_trigger", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_num", type=int, default=32)    # d in equation 2
    parser.add_argument("--frame_interval", type=int, default=2)   # n in equation 1
    parser.add_argument("--num_clips", type=int, default=4)   # 4 clips are needed to evaluate the quality
    parser.add_argument("--segments_num", type=int, default=16)
    parser.add_argument("--resize_size", type=int, default=224)    # the V^{inter} are resized to 224*224
    parser.add_argument("--fragments_size", type=int, default=224)
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['K'], # K: KoNViD-1k  N: LIVE-VQC  Y: YouTube-UGC  Q: LSVQ
                        help="trained datasets (default: ['K', 'N', 'Y', 'Q'])")

    return parser.parse_args()

def main():
    config = parse_config()
    if config.trained_datasets[0] == 'K':
        max_score = 4.64
        yml_name = 'konvid'
    if config.trained_datasets[0] == 'N':
        max_score = 94.2865
        yml_name = 'livevqc'
    if config.trained_datasets[0] == 'Y':
        max_score = 4.698
        yml_name = 'youtube'
    if config.trained_datasets[0] == 'Q':
        max_score = 91.4194
        yml_name = 'lsvq'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/Attack/{}.yml".format(yml_name), help="the option file"
    )
    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.train()
    for k, v in model.named_parameters():
        v.requires_grad = False
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                module.training = False

    for key in opt["data"].keys():
        if "val" not in key and "test" not in key:
            continue
        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"] + "_Test_" + key,
            reinit=True,
        )
        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
        do_attack(val_loader,model,device, opt["data"][key]["args"],max_score,config)
        run.finish()


method_folder = 'SecureVQA'
file_name = os.path.join(method_folder + '_white/')

log_path = 'logs/{}.txt'.format(('White-SecureVQA'))
def log(str_to_log):
    print(str_to_log)
    if not log_path is None:
        with open(log_path, 'a') as f:
            f.write(str_to_log + '\n')
            f.flush()

if __name__ == "__main__":
    main()
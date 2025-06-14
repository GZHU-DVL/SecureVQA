import torch
import securevqa.models_train as models
import securevqa.datasets_train as datasets
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from tqdm import tqdm
import math
import wandb
import yaml
import os
from functools import reduce
from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 20240117
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
            torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types = ["fragments"]

def finetune_epoch(ft_loader, model, model_ema,optimizer, scheduler, device, epoch=-1,
                   need_upsampled=True, need_feat=True, need_fused=False, need_separate_sup=False):
    model.train()
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    log('Number of total parameters: {}, tunable parameters: {}, rate: {}'.format(num_total_param, num_param,
                                                                                    num_param / num_total_param))

    for i, data_video in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):

        Inter_embeddings = data_video[1]
        data = data_video[0]
        optimizer.zero_grad()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)

        if need_upsampled:
            up_video = {}
            for key in sample_types:
                if key + "_up" in data:
                    up_video[key] = data[key + "_up"].to(device)

        y = data["gt_label"].float().detach().to(device).unsqueeze(-1)
        if need_feat:
            scores, feats = model(video, inference=False,
                                  return_pooled_feats=True,
                                  reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            y_pred = y_pred.mean((-3, -2, -1))
        else:
            scores = model(video,Inter_embeddings,inference=False,
                           reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            y_pred = y_pred.mean((-3, -2, -1))
        if need_upsampled:
            if need_feat:
                scores_up, feats_up = model(up_video, inference=False,
                                            return_pooled_feats=True,
                                            reduce_scores=False)
                if len(scores) > 1:
                    y_pred_up = reduce(lambda x, y: x + y, scores_up)
                else:
                    y_pred_up = scores_up[0]
                y_pred_up = y_pred_up.mean((-3, -2, -1))
            else:
                y_pred_up = model(up_video, inference=False).mean((-3, -2, -1))
        # Plain Supervised Loss
        p_loss, r_loss = plcc_loss(y_pred, y), rank_loss(y_pred, y)

        loss =  p_loss + 0.3 * r_loss
        wandb.log(
            {
                "train/plcc_loss": p_loss.item(),
                "train/rank_loss": r_loss.item(),
            }
        )

        if need_separate_sup:
            p_loss_a = plcc_loss(scores[0].mean((-3, -2, -1)), y)
            p_loss_b = plcc_loss(scores[1].mean((-3, -2, -1)), y)
            loss += 0.15 * (p_loss_a + p_loss_b)
            wandb.log(
                {
                    "train/plcc_loss_a": p_loss_a.item(),
                    "train/plcc_loss_b": p_loss_b.item(),
                }
            )
        if need_upsampled:
            ## Supervised Loss for Upsampled Samples
            if need_separate_sup:
                p_loss_up_a = plcc_loss(scores_up[0].mean((-3, -2, -1)), y)
                p_loss_up_b = plcc_loss(scores_up[1].mean((-3, -2, -1)), y)
                loss += 0.15 * (p_loss_up_a + p_loss_up_b)
                wandb.log(
                    {
                        "train/plcc_loss_up_a": p_loss_up_a.item(),
                        "train/plcc_loss_up_b": p_loss_up_b.item(),
                    }
                )
            p_loss_up, r_loss_up = plcc_loss(y_pred_up, y), rank_loss(y_pred_up, y)
            loss += p_loss_up + 0.1 * r_loss_up
            wandb.log(
                {
                    "train/up_plcc_loss": p_loss_up.item(),
                    "train/up_rank_loss": r_loss_up.item(),
                }
            )

            if need_fused:
                fused_mask = torch.where(torch.randn(*y_pred.shape) > 0, 1, 0).to(y_pred.device)
                y_pred_f = y_pred * fused_mask + y_pred_up * (1 - fused_mask)
                p_loss_f, r_loss_f = plcc_loss(y_pred_f, y), rank_loss(y_pred_f, y)
                loss += 0.25 * (p_loss_f + 0.1 * r_loss_f)
                wandb.log(
                    {
                        "train/f_plcc_loss": p_loss_f.item(),
                        "train/f_rank_loss": r_loss_f.item(),
                    }
                )

            if need_feat:
                ## Self-Supervised Loss, Similarity between different sampling densities
                for key in feats:
                    sim_loss = self_similarity_loss(feats[key], feats_up[key])
                    loss += 0.25 * sim_loss
                    wandb.log({f"train/{key}_sim_loss": sim_loss.item(), })

        wandb.log({"train/total_loss": loss.item(), })

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ft_loader.dataset.refresh_hypers()

        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )
    model.eval()


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device).unsqueeze(0)
    with torch.no_grad():
        flops, params = profile(model, (video,))
    print(f"The FLOps of the Variant is {flops / 1e9:.1f}G, with Params {params / 1e6:.2f}M.")


def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', save_name="divide"):
    results = []

    best_s, best_p, best_k, best_r = best_

    for i, data_video in enumerate(tqdm(inf_loader, desc="Validating")):
        Inter_embeddings = data_video[1].repeat(4, 1, 1, 1)
        # it needs to be evaluated four clips during testing, the inter_embeddings are replicated four times to keep the dimensions consistent.
        data = data_video[0]
        result = dict()
        video, video_up = {}, {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                                w).permute(0, 2, 1, 3, 4, 5).reshape(b * data["num_clips"][key], c,
                                                                                     t // data["num_clips"][key], h, w)
            if key + "_up" in data:
                video_up[key] = data[key + "_up"].to(device)
                ## Reshape into clips
                b, c, t, h, w = video_up[key].shape
                video_up[key] = video_up[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                                      w).permute(0, 2, 1, 3, 4, 5).reshape(b * data["num_clips"][key],
                                                                                           c,
                                                                                           t // data["num_clips"][key],
                                                                                           h, w)
        with torch.no_grad():
            result["pr_labels"] = model(video,Inter_embeddings).cpu().numpy()
            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del video, video_up

        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"][:]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    wandb.log({f"val_{suffix}/SRCC-{suffix}": s, f"val_{suffix}/PLCC-{suffix}": p, f"val_{suffix}/KRCC-{suffix}": k,
               f"val_{suffix}/RMSE-{suffix}": r})

    log("val_{}/SRCC-{} = {}".format(suffix, suffix, s))
    log("val_{}/PLCC-{} = {}".format(suffix, suffix, p))
    log("val_{}/KRCC-{} = {}".format(suffix, suffix, k))
    log("val_{}/RMSE-{} = {}".format(suffix, suffix, r))

    if "pr_labels_up" in results[0]:
        pr_labels_up = [np.mean(r["pr_labels_up"][:]) for r in results]
        pr_labels_up = rescale(pr_labels_up, gt_labels)

        ups = spearmanr(gt_labels, pr_labels_up)[0]
        upp = pearsonr(gt_labels, pr_labels_up)[0]
        upk = kendallr(gt_labels, pr_labels_up)[0]
        upr = np.sqrt(((gt_labels - pr_labels_up) ** 2).mean())

        wandb.log({f"val_{suffix}/up-SRCC-{suffix}": ups, f"val_{suffix}/up-PLCC-{suffix}": upp,
                   f"val_{suffix}/up-KRCC-{suffix}": upk, f"val_{suffix}/up-RMSE-{suffix}": upr})
        log("val_{}/up-SRCC-{} = {}".format(suffix, suffix, ups))
        log("val_{}/up-PLCC-{} = {}".format(suffix, suffix, upp))
        log("val_{}/up-KRCC-{} = {}".format(suffix, suffix, upk))
        log("val_{}/up-RMSE-{} = {}".format(suffix, suffix, upr))
    del results, result  # , video, video_up
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"pretrained_weights/{save_name}_{suffix}_dev_v0.0.pth",
        )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    wandb.log(
        {
            f"val_{suffix}/best_SRCC-{suffix}": best_s,
            f"val_{suffix}/best_PLCC-{suffix}": best_p,
            f"val_{suffix}/best_KRCC-{suffix}": best_k,
            f"val_{suffix}/best_RMSE-{suffix}": best_r,
        }
    )
    log("val_{}/best_SRCC-{} = {}".format(suffix, suffix, best_s))
    log("val_{}/best_PLCC-{} = {}".format(suffix, suffix, best_p))
    log("val_{}/best_KRCC-{} = {}".format(suffix, suffix, best_k))
    log("val_{}/best_RMSE-{} = {}".format(suffix, suffix, best_r))

    log("For {} videos, \nthe accuracy of the model: {} is as follows:\n  SROCC: {} best: {} \n  PLCC:  {} best: {}  \n  KROCC: {} best: {} \n  RMSE:  {} best: {}."
        .format(len(inf_loader), suffix, s, best_s, p, best_p, k, best_k, r, best_k))
    return best_s, best_p, best_k, best_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/Training/train.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## defining model and loading checkpoint


    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1

    for split in range(num_splits):

        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("val"):
                val_datasets[key] = getattr(datasets,
                                            opt["data"][key]["type"])(opt["data"][key]["args"],key)

        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt["num_workers"],
            )

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"],key)
                train_datasets[key] = train_dataset

        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
            )

        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"] + f'_{split}' if num_splits > 1 else opt["name"],
            reinit=True,
        )

        if "load_path_aux" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)["state_dict"]
            aux_state_dict = torch.load(opt["load_path_aux"], map_location=device)["state_dict"]

            from collections import OrderedDict

            fusion_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "head" in k:
                    continue
                if k.startswith("vqa_head"):
                    ki = k.replace("vqa", "fragments")
                else:
                    ki = k
                fusion_state_dict[ki] = v

            for k, v in aux_state_dict.items():
                if "head" in k:
                    continue
                if k.startswith("frag"):
                    continue
                if k.startswith("vqa_head"):
                    ki = k.replace("vqa", "resize")
                else:
                    ki = k
                fusion_state_dict[ki] = v
            state_dict = fusion_state_dict
            print(model.load_state_dict(state_dict))

        elif "load_path" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)

            if "state_dict" in state_dict:
                ### migrate training weights from mmaction
                state_dict = state_dict["state_dict"]
                from collections import OrderedDict

                i_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if "head" in key:
                        continue
                    if "cls" in key:
                        tkey = key.replace("cls", "vqa")
                    elif "backbone" in key:
                        i_state_dict[key] = state_dict[key]
                        i_state_dict["fragments_" + key] = state_dict[key]
                        i_state_dict["resize_" + key] = state_dict[key]
                    else:
                        i_state_dict[key] = state_dict[key]
            t_state_dict = model.state_dict()
            for key, value in t_state_dict.items():
                if key in i_state_dict and i_state_dict[key].shape != value.shape:
                    i_state_dict.pop(key)

            print(model.load_state_dict(i_state_dict, strict=False))


        if opt["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        param_groups = []

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [
                    {"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]


        optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                      weight_decay=opt["optimizer"]["wd"],
                                      )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
            bests_n[key] = -1, -1, -1, 1000

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = False

        for epoch in range(opt["l_num_epochs"]):
            print(f"Linear Epoch {epoch}:")
            log("#####\n Epoch = {}".format(epoch))
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema,optimizer, scheduler, device, epoch,
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key + "_s",
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix=key + '_n',
                    )
                else:
                    bests_n[key] = bests[key]

        if opt["l_num_epochs"] >= 0:
            for key in val_loaders:
                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

                print(
                    f"""For the linear transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}
                    RMSE:  {bests_n[key][3]:.4f}."""
                )

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = True

        for epoch in range(opt["num_epochs"]):
            log("#\n Epoch = {}".format(epoch))
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema,optimizer, scheduler, device, epoch,
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    val_loaders[key],
                    model_ema if model_ema is not None else model,
                    device, bests[key], save_model=opt["save_model"], save_name=opt["name"],
                    suffix=key + "_s",
                )
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix=key + '_n',
                    )
                else:
                    bests_n[key] = bests[key]

        if opt["num_epochs"] > 0:
            for key in val_loaders:
                print(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

                print(
                    f"""For the finetuning process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-n is as follows:
                    SROCC: {bests_n[key][0]:.4f}
                    PLCC:  {bests_n[key][1]:.4f}
                    KROCC: {bests_n[key][2]:.4f}
                    RMSE:  {bests_n[key][3]:.4f}."""
                )

        run.finish()

log_path = 'logs/{}.txt'.format(('SecureVQA'))
def log(str_to_log):
    print(str_to_log)
    if not log_path is None:
        with open(log_path, 'a') as f:
            f.write(str_to_log + '\n')
            f.flush()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

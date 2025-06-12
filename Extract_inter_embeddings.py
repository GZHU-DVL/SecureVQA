import argparse
import os
import numpy as np
import torch
import yaml
import securevqa.datasets as datasets
from torchvision import transforms
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 20240107
torch.manual_seed(seed)
np.random.seed(seed)

##### For convenience, we use dataloader to extract the inter-frame information, so the batch size can only be set to 1!!!!
def main(config):

    with open(config.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    extract_datasets = {}
    for key in opt["data"]:
        if key.startswith("train") or key.startswith("val"):
            extract_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])
            extract_datasets[key] = extract_dataset
    extract_loaders = {}
    for key, extract_dataset in extract_datasets.items():
        extract_loaders[key] = torch.utils.data.DataLoader(
            extract_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
        )

    with torch.no_grad():
        for key, extract_loader in extract_loaders.items():
            feature_save_folder = 'Inter_embeddings_{}/'.format(key)
            if not os.path.exists(feature_save_folder):
                os.makedirs(feature_save_folder)
            for i, data in enumerate(extract_loader):
                Inter_embeddings = data[0]    # E^{inter}
                video_name = data[1][0][:-4]
                print(i, video_name,Inter_embeddings.shape)
                np.save(feature_save_folder + video_name + '_inter_embeddings',
                        Inter_embeddings.to('cpu').numpy())


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/Embeddings/inter.yml", help="the option file"
    )
    config = parser.parse_args()
    main(config)



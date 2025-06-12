## SecureVQA
Official Repository for "SecureVQA: Robust Video Quality Assessment Resisting Adversarial Attacks"

# Usage
## Requirements
The original library is build on 
* python==3.8.8
* torch==1.12.1
* detectron2==0.6
* scikit-video==1.1.11
* scikit-image==0.19.1
  
To get all the requirements, please run

```
pip install -r requirements.txt
```

## Dataset Preparation
### VQA Datasets

The experiments are conducted on four mainstream video datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [YouTube-UGC](https://media.withyoutube.com/), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. 

## Train the SecureVQA 
### Get Pretrained Weights
Download the [Swin-Transformer Weights](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth) pretrained on kinetics to initialize the model. 

### Extract Inter-Frame Embeddings
First, you need to download the dataset, and copy their local addresses to "data_prefix" of ./options/Embeddings/inter.yml. 

```
python Extract_inter_embeddings.py
```
***Note:***  For convenience, we use dataloader to extract the inter-frame information, so the batch size can only be set to 1!!!!

### Train on the Large-Scale Dataset LSVQ

First, you need to download the dataset, and copy their local addresses to "data_prefix" of ./options/Training/train.yml. 

```
python main.py
```

### Finetune on the Small-Scale Dataset

Similarly, you need to copy the local addresses to "data_prefix" of ./options/Finetune/**.yml. Each dataset is randomly splited 10 times for evaluation.
```
python Finetune.py
```

## The security of SecureVQA 

In the experiments to verify the security of SecureVQA, the following points should be noted:

* Since the attack videos need to be selected from the testing set, the attack videos in ./examplar_data_labels/ need to be adjusted for different training models.
* Because the efficiency of existing attacks is too low, in the test process, we assume that the attacker knows which frames SecureVQA selects for evaluation, and only the frames selected by SecureVQA are attacked. This is advantageous to the attacker.
*  If you want to further improve the efficiency, the FasterVQA strategy can be adopted to select only one clip to evaluate the quality. This can quickly verify the safety of the model. 

The predicted scores for original videos and corresponding adversarial videos.
![image](Deffense_effect.jpg)

### Under White-Box Setting
First, you need to copy the local address of the trained SecureVQA model into "test_load_path" of ./options/Attack/***.yml.
```
python SecureVQA_under_white_box.py
```

### Under Black-Box Setting
Similarly, you need to copy the local address of the trained SecureVQA model into "test_load_path" of ./options/Attack/***.yml.
```
python SecureVQA_under_black_box.py
```

## License
This source code is made available for research purpose only.

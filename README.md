# README of XAISEUNG 

## 1. 준비

cuda 11.4 + cudnn 8.2.4 환경에서 테스트하여 작동을 확인했습니다.

### 1. Clone this repository
  ```bash
  git clone https://github.com/xaiseung/ghost.git
  cd ghost
  git submodule init
  git submodule update
  ```
### 2. Install dependent packages
   
환경 구축 방식은 2가지가 있습니다.

**2-1. conda 사용**

  가상환경에 cudatoolkit으로 11.4를 설치합니다.

  이때 가상환경의 cuda의 위치를 확인하신 뒤 cudnn 8.2.4를 직접 설치하셔야 합니다

  이것이 번거로우시다면 환경 구축 후 cudatoolkit을 지우고 컴퓨터에 설치된 cuda를 이용하시기 바랍니다.
  ```bash
  conda env create -n ghost_cu114 -f cond_req_cuda114.yaml
  ```

**2-2. pip 사용 (cuda 11.4와 cudnn 8.2.4는 설치했다고 가정)**

  - 가상 환경 생성 후 다음을 실행

   ```bash
   pip install -r requirements_cu114
   ```
  
### 3. Download weights
  ```bash
  sh download_models.sh
  ```

### 4. (선택) 데이터셋 다운로드 및 전처리
- [VggFace2](https://www.kaggle.com/datasets/dimarodionov/vggface2) 다운로드
- 전처리
```bash
  python preprocess_vgg.py --path_to_dataset {PATH_TO_DATASET} --save_path {SAVE_PATH}
```


## 2. 코드 파일 설명

각 실행 과정을 이해하고 추가적인 아이디어(PCA 분석 기반 변조)를 붙이기 위해서 다음 .ipynb 파일들을 만들었습니다.

이해 과정은 원본 깃허브의 `SberSwapInference.ipynb`와 소스코드 내부를 살펴보며 작성했습니다.

### 실행과정 이해
- `[1]swap_demo.ipynb`
  - 이미지 또는 영상에 얼굴을 치환하고 저장하는 간단한 예제입니다.
- `[2-1]about_5keypoints.ipynb`
  - 얼굴 크롭 및 정렬에 사용하는 5-key points 시각화하는 예제입니다.
- `[2-2]about_mask_N_roll_out_process.ipynb`
  - source로 치환된 얼굴을 target으로 붙일 때 마스크를 사용하는 과정을 시각화하고
  - 전체 과정을 풀어 써보고 작동해보는 예제입니다.

### 아이디어 (PCA 기반 변조)
- 전처리된 [VggFace2](https://www.kaggle.com/datasets/dimarodionov/vggface2)가 필요합니다.
- `[3-1]extract_src_emb_and_pca.ipynb`
  - src 이미지의 임베딩을 추출하고 pca 분석 모델을 저장합니다.
- `[3-2]extract_GAN_tgt_zattr.ipynb`
  - GAN (U-net)에서 tgt을 입력으로 한 중간 zattr를 추출하여 저장합니다.
- `[3-3]PCA_GANtgt.ipynb`
  - 3-2에서 추출한 중간 zattr 값으로 PCA 분석 모델을 만들고 저장합니다.
- `[3-3B]PCA_GANtgt_alter.ipynb`
  - 3-2에서 추출한 중간 zattr 값으로 PCA 분석 모델을 만들고 저장합니다.
  - 이전 코드와 PCA 파라미터가 조금 다릅니다.
  - doublePCA (각 레이어의 PCA 벡터들을 이어붙인 뒤 다시 PCA)를 위한 코드입니다.
- `[4-1]modify_src_emb_thru_pca.ipynb`
  - PCA 모델을 바탕으로 src_emb을 조절하며 결과를 확인하고 그림을 그립니다.
- `[4-2]modify_zattr_emb_thru_pca.ipynb`
  - PCA 모델로 zattr emb를 조절하여 결과를 확인하고 그림을 그립니다.
- `[4-2B]modify_zattr_emb_thru_doublepca.ipynb`
  - DoublePCA 모델로 zattr emb를 조절하여 결과를 확인하고 그림을 그립니다.
- `[5]smile_application.ipynb`
  - 얼굴을 치환하고 target의 zattr를 건드려 웃게 만듭니다.
  - TODO: 영상 입력 업로드로 받기 


### 이하 원본 GHOST 리파지토리의 README.md


[[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9851423)] [[Habr](https://habr.com/ru/company/sberbank/blog/645919/)]

# 👻 GHOST: Generative High-fidelity One Shot Transfer 

Our paper ["GHOST—A New Face Swap Approach for Image and Video Domains"](https://ieeexplore.ieee.org/abstract/document/9851423) has been published on IEEE Xplore.

<p align="left">
  Google Colab Demo
</p>
<p align="left">
  <a href="https://colab.research.google.com/drive/1vXTpsENipTmjTMggwveCkXASwxUk270n">
  <img src="https://colab.research.google.com/assets/colab-badge.svg"/>
  </a>
</p>

## GHOST Ethics 

Deepfake stands for a face swapping algorithm where the source and target can be an image or a video. Researchers have investigated sophisticated generative adversarial networks (GAN), autoencoders, and other approaches to establish precise and robust algorithms for face swapping. However, the achieved results are far from perfect in terms of human and visual evaluation. In this study, we propose a new one-shot pipeline for image-to-image and image-to-video face swap solutions - GHOST (Generative High-fidelity One Shot Transfer).

Deep fake synthesis methods have been improved a lot in quality in recent years. The research solutions were wrapped in easy-to-use API, software and different plugins for people with a little technical knowledge. As a result, almost anyone is able to make a deepfake image or video by just doing a short list of simple operations. At the same time, a lot of people with malicious intent are able to use this technology in order to produce harmful content. High distribution of such a content over the web leads to caution, disfavor and other negative feedback to deepfake synthesis or face swap research.

As a group of researchers, we are not trying to denigrate celebrities and statesmen or to demean anyone. We are computer vision researchers, we are engineers, we are activists, we are hobbyists, we are human beings. To this end, we feel that it's time to come out with a standard statement of what this technology is and isn't as far as us researchers are concerned.
* GHOST is not for creating inappropriate content.
* GHOST is not for changing faces without consent or with the intent of hiding its use.
* GHOST is not for any illicit, unethical, or questionable purposes.
* GHOST exists to experiment and discover AI techniques, for social or political commentary, for movies, and for any number of ethical and reasonable uses.

We are very troubled by the fact that GHOST can be used for unethical and disreputable things. However, we support the development of tools and techniques that can be used ethically as well as provide education and experience in AI for anyone who wants to learn it hands-on. Now and further, we take a **zero-tolerance approach** and **total disregard** to anyone using this software for any unethical purposes and will actively discourage any such uses.


## Image Swap Results 
![](/examples/images/example1.png)

![](/examples/images/example2.png)

## Video Swap Results
<div>
<img src="/examples/videos/orig.webp" width="360"/>
<img src="/examples/videos/elon.webp" width="360"/>
<img src="/examples/videos/khabenskii.webp" width="360"/>
<img src="/examples/videos/mark.webp" width="360"/>
<div/>

## Installation
  
1. Clone this repository
  ```bash
  git clone https://github.com/sberbank-ai/sber-swap.git
  cd sber-swap
  git submodule init
  git submodule update
  ```
2. Install dependent packages
  ```bash
  pip install -r requirements.txt
  ```
  If it is not possible to install onnxruntime-gpu, try onnxruntime instead  
  
3. Download weights
  ```bash
  sh download_models.sh
  ```
## Usage
  1. Colab Demo <a href="https://colab.research.google.com/drive/1B-2JoRxZZwrY2eK_E7TB5VYcae3EjQ1f"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> or you can use jupyter notebook [SberSwapInference.ipynb](SberSwapInference.ipynb) locally
  2. Face Swap On Video
  
  Swap to one specific person in the video. You must set face from the target video (for example, crop from any frame).
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE} --target_faces_paths {PATH_TO_IMAGE} --target_video {PATH_TO_VIDEO}
  ```
  Swap to many person in the video. You must set multiple faces for source and the corresponding multiple faces from the target video.
  ```bash
  python inference.py --source_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_faces_paths {PATH_TO_IMAGE PATH_TO_IMAGE ...} --target_video {PATH_TO_VIDEO}
  ```
  3. Face Swap On Image
  
  You may set the target face, and then source will be swapped on this person, or you may skip this parameter, and then source will be swapped on any person in the image.
  ```bash
  python inference.py --target_path {PATH_TO_IMAGE} --image_to_image True
  ```
  
## Training
  
We also provide the training code for face swap model as follows:
  1. Download [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/).
  2. Crop and align faces with out detection model.
  ```bash
  python preprocess_vgg.py --path_to_dataset {PATH_TO_DATASET} --save_path {SAVE_PATH}
  ```
  3. Start training. 
  ```bash
  python train.py --run_name {YOUR_RUN_NAME}
  ```
We provide a lot of different options for the training. More info about each option you can find in `train.py` file. If you would like to use wandb logging of the experiments, you should login to wandb first  `--wandb login`.
  
### Tips
  1. For the first epochs we suggest not to use eye detection loss and scheduler if you train from scratch.
  2. In case of finetuning you can variate losses coefficients to make the output look similar to the source identity, or vice versa, to save features and attributes of target face.
  3. You can change the backbone of the attribute encoder and num_blocks of AAD ResBlk using parameters `--backbone` and `--num_blocks`.
  4. During the finetuning stage you can use our pretrain weights for generator and discriminator that are located in `weights` folder. We provide the weights for models with U-Net backbone and 1-3 blocks in AAD ResBlk. The main model architecture contains 2 blocks in AAD ResBlk.
  
## Cite
If you use our model in your research, we would appreciate using the following citation

  ### BibTeX Citation
  ```
  @article{9851423,  
           author={Groshev, Alexander and Maltseva, Anastasia and Chesakov, Daniil and Kuznetsov, Andrey and Dimitrov, Denis},  
           journal={IEEE Access},   
           title={GHOST—A New Face Swap Approach for Image and Video Domains},   
           year={2022},  
           volume={10},  
           number={},  
           pages={83452-83462},  
           doi={10.1109/ACCESS.2022.3196668}
  }
  ```
  
  ### General Citation
  
  A. Groshev, A. Maltseva, D. Chesakov, A. Kuznetsov and D. Dimitrov, "GHOST—A New Face Swap Approach for Image and Video Domains," in IEEE Access, vol. 10, pp. 83452-83462, 2022, doi: 10.1109/ACCESS.2022.3196668.
  

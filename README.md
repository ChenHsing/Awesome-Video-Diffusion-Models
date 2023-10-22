

# [A Survey on Video Diffusion Models](https://arxiv.org/abs/2310.10647) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.10647) 


<div style="text-align:center; font-size: 18px;">
    <p>
    <a href="https://chenhsing.github.io">Zhen Xing</a>, 
    Qijun Feng,
    Haoran Chen,
    <a href="https://scholar.google.com/citations?user=NSJY12IAAAAJ&hl=zh-CN" >Qi Dai,</a>
    <a href="https://scholar.google.com/citations?user=Jkss014AAAAJ&hl=zh-CN&oi=ao" >Han Hu,</a>
    <a href="https://scholar.google.com/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao" >Hang Xu,</a>
    <a href="https://scholar.google.com/citations?user=7t12hVkAAAAJ&hl=en" >Zuxuan Wu,</a>
    <a href="https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en" >Yu-Gang Jiang </a>
     </p>
</div>




<p align="center">
<img src="asset/fish.webp" width="160px"/>
<img src="asset/tree.gif" width="160px"/>    
<img src="asset/raccoon.gif" width="160px"/>    
</p>


<p align="center">
<img src="asset/fly.gif" width="240px"/>  
<img src="asset/fly3.gif" width="240px"/>  
</p>


<p align="center">
<img src="asset/1.gif" width="120px"/>
<img src="asset/2.gif" width="120px"/>
<img src="asset/3.gif" width="120px"/>
<img src="asset/4.gif" width="120px"/>
</p>


<p align="center">
(Source: <a href="https://makeavideo.studio/">Make-A-Video</a>, <a href="https://chenhsing.github.io/SimDA/">SimDA</a>, <a href="https://research.nvidia.com/labs/dir/pyoco/">PYoCo</a>,
<a href="https://research.nvidia.com/labs/toronto-ai/VideoLDM/">Video LDM</a> and <a href="https://tuneavideo.github.io/">Tune-A-Video</a>)
</p>



- [News] The Chinese translation is available on [Zhihu](https://zhuanlan.zhihu.com/p/661860981). Special thanks to [Dai-Wenxun](https://github.com/Dai-Wenxun) for this.


## Open-source Toolboxes and Foundation Models 

| Methods | Task | Github|
|:-----:|:-----:|:-----:|
| [GEN-2](https://app.runwayml.com/)   | T2V Generation & Editing | - |
| [ModelScope](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)  | T2V Generation | [![Star](https://img.shields.io/github/stars/modelscope/modelscope.svg?style=social&label=Star)](https://github.com/modelscope/modelscope) |
| [ZeroScope](https://easywithai.com/ai-video-generators/zeroscope/)  | T2V Generation | -|
| [T2V Synthesis Colab](https://github.com/camenduru/text-to-video-synthesis-colab) | T2V Genetation |[![Star](https://img.shields.io/github/stars/camenduru/text-to-video-synthesis-colab.svg?style=social&label=Star)](https://github.com/camenduru/text-to-video-synthesis-colab)|
| [VideoCraft](https://github.com/VideoCrafter/VideoCrafter) | T2V Genetation & Editing |[![Star](https://img.shields.io/github/stars/VideoCrafter/VideoCrafter.svg?style=social&label=Star)](https://github.com/VideoCrafter/VideoCrafter)|
| [Diffusers (T2V synthesis)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/text_to_video#texttovideo-synthesis) | T2V Genetation |-|
| [AnimateDiff](https://github.com/guoyww/AnimateDiff) | Personalized T2V Genetation |[![Star](https://img.shields.io/github/stars/guoyww/AnimateDiff.svg?style=social&label=Star)](https://github.com/guoyww/AnimateDiff)|
| [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) |  T2V Genetation |[![Star](https://img.shields.io/github/stars/Picsart-AI-Research/Text2Video-Zero.svg?style=social&label=Star)](https://github.com/Picsart-AI-Research/Text2Video-Zero)|
| [HotShot-XL](https://github.com/hotshotco/Hotshot-XL) |  T2V Genetation |[![Star](https://img.shields.io/github/stars/hotshotco/Hotshot-XL.svg?style=social&label=Star)](https://github.com/hotshotco/Hotshot-XL)|
| [Genmo](https://www.genmo.ai/) |  T2V Genetation |-|
| [Fliki](https://fliki.ai/) | T2V Generation | -|



## Table of Contents 

- [Video Generation](#video-generation)
- - [Data](#data)
- - - [Caption-level](#caption-level)
- - - [Category-level](#category-level)
- - [T2V Generation](#text-to-video-generation)
- - - [Training-based](#training-based)
- - - [Training-free](#training-free)
- - [Video Generation with other Condtions](#modality-control-video-generation)
- - - [Pose-gudied](#pose-guided-video-generation)
- - - [Instruct-guided](#instruct-guided-video-generation)
- - - [Sound-guided](#sound-guided-video-generation)
- - - [Brain-guided](#brain-guided-video-generation)
- - - [Multi-Modal guided](#multi-modal-guided-video-generation)
- - [Unconditional Video Generation](#domain-specific-video-generation)
- - - [U-Net based](#category-specific)
- - - [Transformer-based](#personalized-video)
- - [Video Completion](#video-completion)
- - - [Video Enhance and Restoration](#video-enhancement-and-restoration)
- - - [Video Prediction](#video-prediction)
- [Video Editing](#video-editing)
- - [Text guided Video Editing](#text-guided-video-editing)
- - - [Training-based Editing](#general-editing-model)
- - - [One-shot Editing](#one-shot-editing-model)
- - - [Traning-free](#training-free-editing-model)
- - [Modality-guided Video Editing](#modality-guided)
- - - [Motion-guided](#motion-guided-editing-model)
- - - [Instruct-guided](#instruct-guided-editing-model)
- - - [Sound-guided](#sound-guided-editing-model)
- - - [Multi-Modal Control](#multi-modal-control-editing-model)
- - [Domain-specific editing](#domain-specific-editing-model)
- - [Non-diffusion editing](#non-diffusion-editing-Model)
- [Video Understanding](#video-understanding)
- [Contact](#Contact)





# Video Generation

## Data 

### Caption-level

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
|[CelebV-Text: A Large-Scale Facial Text-Video Dataset](https://arxiv.org/pdf/2303.14717.pdf)|  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2303.14717.pdf) |   [![Star](https://img.shields.io/github/stars/CelebV-Text/CelebV-Text.svg?style=social&label=Star)](https://github.com/CelebV-Text/CelebV-Text) | - | CVPR, 2023
|[InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation](https://arxiv.org/abs/2307.06942) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.06942)|   [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVideo.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)| - | May, 2023 |
|[VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation](https://arxiv.org/abs/2305.10874)| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10874) | - | - | May, 2023|
|[Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions](https://arxiv.org/abs/2111.10337)  |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2111.10337)|- |- |Nov, 2021 |
| [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/abs/2104.00650) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2104.00650)  | - | - |ICCV, 2021 |
|[MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) |  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) | -| -| CVPR, 2016|




### Category-level


| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
|[UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402)|  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1212.0402) |   - | - | Dec., 2012
|[First Order Motion Model for Image Animation](https://arxiv.org/abs/2003.00196) |     [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2003.00196) | -|- | May, 2023 |
|[Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks](https://arxiv.org/abs/1709.07592) |   [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1709.07592) | -| -| CVPR,2018|



### Metric

| Title | arXiv | Github| WebSite | Pub. & Date
|:-----:|:-----:|:-----:|:-----:|:-----:|
|[EvalCrafter: Benchmarking and Evaluating Large Video Generation Models](https://arxiv.org/abs/2310.11440) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.11440)  |   - | - | Oct., 2023 |
|[Measuring the Quality of Text-to-Video Model Outputs: Metrics and Dataset](https://arxiv.org/abs/2309.08009) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.08009)  |   - | - | Sep., 2023 |


## Text-to-Video Generation

### Training-based 

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors](https://arxiv.org/abs/2310.12190) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.12190) | [![Star](https://img.shields.io/github/stars/AILab-CVC/VideoCrafter.svg?style=social&label=Star)](https://github.com/AILab-CVC/VideoCrafter) | -| Oct., 2023 |
|[LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation](https://arxiv.org/abs/2310.10769) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.10769) | [![Star](https://img.shields.io/github/stars/RQ-Wu/LAMP.svg?style=social&label=Star)](https://github.com/RQ-Wu/LAMP) | [![Website](https://img.shields.io/badge/Website-9cf)](https://rq-wu.github.io/projects/LAMP/) | Oct., 2023 |
|[DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model](https://arxiv.org/abs/2310.07771) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.07771) | - | - | Oct, 2023 |
|[MotionDirector: Motion Customization of Text-to-Video Diffusion Models](https://arxiv.org/abs/2310.08465) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.08465)  | [![Star](https://img.shields.io/github/stars/showlab/MotionDirector?style=social)](https://github.com//showlab/MotionDirector) | [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/MotionDirector/) | Oct, 2023 |
|[VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning](https://arxiv.org/abs/2309.15091) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.15091)  |   - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://videodirectorgpt.github.io/) | Sep., 2023 |
|[Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2309.15818) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.15818)  | [![Star](https://img.shields.io/github/stars/showlab/Show-1?style=social)](https://github.com//showlab/Show-1) |   [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/Show-1/) | Sep., 2023 |
|[LaVie: High-Quality Video Generation with Cascaded Latent Diffusion Models](https://arxiv.org/abs/2309.15103) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.15103)  |   - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://vchitect.github.io/LaVie-project/) | Sep., 2023 |
|[Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation](https://arxiv.org/abs/2309.03549) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.03549)  |   - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://arxiv.org/abs/2309.03549) | Sep., 2023 |
|[VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation](https://arxiv.org/abs/2309.00398) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.00398)  |   - |   - | Sep., 2023 |
|[Text2Performer: Text-Driven Human Video Generation](https://arxiv.org/abs/2304.08483) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.08483) | [![Star](https://img.shields.io/github/stars/yumingj/Text2Performer.svg?style=social&label=Star)](https://github.com/yumingj/Text2Performer) | [![Website](https://img.shields.io/badge/Website-9cf)](https://yumingj.github.io/projects/Text2Performer) | Apr., 2023 |
|[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.04725) | [![Star](https://img.shields.io/github/stars/guoyww/animatediff.svg?style=social&label=Star)](https://github.com/guoyww/animatediff/) | [![Website](https://img.shields.io/badge/Website-9cf)](https://animatediff.github.io/) | Jul., 2023 |
|[Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models](https://arxiv.org/abs/2308.13812) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.13812)  |   - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://haofei.vip/Dysen-VDM/) | Aug., 2023 |
|[SimDA: Simple Diffusion Adapter for Efficient Video Generation](https://arxiv.org/abs/2308.09710) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.09710)  | [![Star](https://img.shields.io/github/stars/ChenHsing/SimDA.svg?style=social&label=Star)](https://github.com/ChenHsing/SimDA)  |   [![Website](https://img.shields.io/badge/Website-9cf)](https://chenhsing.github.io/SimDA/) | Aug., 2023 |
|[Dual-Stream Diffusion Net for Text-to-Video Generation](https://arxiv.org/abs/2308.08316) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.08316)  |   - |   - | Aug., 2023 |
|[ModelScope Text-to-Video Technical Report](https://arxiv.org/abs/2308.06571) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.06571)  |   - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) | Aug., 2023 |
|[InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation](https://arxiv.org/abs/2307.06942) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.06942)  | [![Star](https://img.shields.io/github/stars/OpenGVLab/InternVideo.svg?style=social&label=Star)](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)  | - | Jul., 2023 |
|[VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation](https://arxiv.org/abs/2305.10874) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10874) | - | - | May, 2023 |
|[Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models](https://arxiv.org/abs/2305.10474) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.10474) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/dir/pyoco/) | May, 2023 |
|[Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.08818) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://research.nvidia.com/labs/toronto-ai/VideoLDM/) | - | CVPR 2023 |
|[Latent-Shift: Latent Diffusion with Temporal Shift](https://arxiv.org/abs/2304.08477) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.08477) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://latent-shift.github.io/) | - | Apr., 2023 |
|[Probabilistic Adaptation of Text-to-Video Models](https://arxiv.org/abs/2306.01872) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01872) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://video-adapter.github.io/video-adapter/) | Jun., 2023 |
|[NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation](https://arxiv.org/abs/2303.12346) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12346) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://msra-nuwa.azurewebsites.net/#/) | Mar., 2023 |
|[ED-T2V: An Efficient Training Framework for Diffusion-based Text-to-Video Generation](https://ieeexplore.ieee.org/abstract/document/10191565) | - | - | - | IJCNN, 2023 |
|[MagicVideo: Efficient Video Generation With Latent Diffusion Models](https://arxiv.org/abs/2211.11018) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11018) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://magicvideo.github.io/#) | - | Nov., 2022 |
|[Imagen Video: High Definition Video Generation With Diffusion Models](https://arxiv.org/abs/2210.02303) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.02303) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://imagen.research.google/video/) | - | Oct., 2022 |
|[VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation](https://arxiv.org/abs/2303.08320) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.08320) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) | - | CVPR 2023 |
|[Make-A-Video: Text-to-Video Generation without Text-Video Data](https://openreview.net/forum?id=nJfylDvgzlq) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://openreview.net/forum?id=nJfylDvgzlq) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://makeavideo.studio) | - | ICLR 2023 |
|[Latent Video Diffusion Models for High-Fidelity Video Generation With Arbitrary Lengths](https://arxiv.org/abs/2211.13221) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.13221) | [![Star](https://img.shields.io/github/stars/YingqingHe/LVDM.svg?style=social&label=Star)](https://github.com/YingqingHe/LVDM) |   [![Website](https://img.shields.io/badge/Website-9cf)](https://yingqinghe.github.io/LVDM/) | Nov., 2022 |
|[Video Diffusion Models](https://arxiv.org/abs/2204.03458) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2204.03458) | - |   [![Website](https://img.shields.io/badge/Website-9cf)](https://video-diffusion.github.io/) | - | Apr., 2022 |





### Training-free

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation](https://arxiv.org/abs/2310.07697) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.07697) | [![Star](https://img.shields.io/github/stars/pengbo807/ConditionVideo.svg?style=social&label=Star)](https://github.com/pengbo807/ConditionVideo)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://pengbo807.github.io/conditionvideo-website/) | Oct, 2023 |
|[LLM-grounded Video Diffusion Models](https://arxiv.org/abs/2309.17444) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.17444) | - | - | Oct, 2023 |
|[Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator](https://arxiv.org/abs/2309.14494) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.14494) | [![Star](https://img.shields.io/github/stars/SooLab/Free-Bloom.svg?style=social&label=Star)](https://github.com/SooLab/Free-Bloom) | - | NeurIPS, 2023 |
|[DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/abs/2308.03463) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.03463) | - | - | Aug, 2023 |
|[Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation](https://arxiv.org/abs/2305.14330) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.14330) | [![Star](https://img.shields.io/github/stars/KU-CVLAB/DirecT2V.svg?style=social&label=Star)](https://github.com/KU-CVLAB/DirecT2V) | - | May, 2023 |
|[Text2video-Zero: Text-to-Image Diffusion Models Are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13439) | [![Star](https://img.shields.io/github/stars/Picsart-AI-Research/Text2Video-Zero.svg?style=social&label=Star)](https://github.com/Picsart-AI-Research/Text2Video-Zero) | [![Website](https://img.shields.io/badge/Website-9cf)](https://text2video-zero.github.io/) | Mar., 2023 |






## Video Generation with other conditions

### Pose-guided Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[DisCo: Disentangled Control for Referring Human Dance Generation in Real World](https://arxiv.org/abs/2307.00040) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.00040) | [![Star](https://img.shields.io/github/stars/Wangt-CN/DisCo.svg?style=social&label=Star)](https://github.com/Wangt-CN/DisCo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://disco-dance.github.io/) | Jul., 2023 |
|[Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model](https://arxiv.org/pdf/2308.07749.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.07749.pdf) | - | - | Aug., 2023 |
|[DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion](https://arxiv.org/abs/2304.06025) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06025) | [![Star](https://img.shields.io/github/stars/johannakarras/DreamPose.svg?style=social&label=Star)](https://github.com/johannakarras/DreamPose) | [![Website](https://img.shields.io/badge/Website-9cf)](https://grail.cs.washington.edu/projects/dreampose/) | Apr., 2023 |
|[Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos](https://arxiv.org/abs/2304.01186) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.01186) | [![Star](https://img.shields.io/github/stars/mayuelala/FollowYourPose.svg?style=social&label=Star)](https://github.com/mayuelala/FollowYourPose) | [![Website](https://img.shields.io/badge/Website-9cf)](https://follow-your-pose.github.io/) | Apr., 2023 |


### Motion-guided  Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Motion-Conditioned Diffusion Model for Controllable Video Synthesis](https://arxiv.org/abs/2304.14404) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.14404) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://tsaishien-chen.github.io/MCDiff/) | Apr., 2023 |
|[DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory](https://arxiv.org/abs/2308.08089) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.08089) | - | - | Aug., 2023 |


### Sound-guided Video Generation
| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion](https://arxiv.org/abs/2309.04509) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.04509) | - | - | ICCV, 2023 |
|[Generative Disco: Text-to-Video Generation for Music Visualization](https://arxiv.org/abs/2304.08551) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.08551) | - | - | Apr., 2023 |
| [AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion](https://arxiv.org/abs/2305.04001) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.04001) | - | - | CVPRW, 2023 |

### Image-guided Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Make-It-4D: Synthesizing a Consistent Long-Term Dynamic Scene Video from a Single Image](https://arxiv.org/abs/2308.10257) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.10257) | - | - | MM, 2023 |
|[Generative Image Dynamics](https://arxiv.org/abs/2309.07906) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.07906) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://generative-dynamics.github.io/) | Sep., 2023 |
|[LaMD: Latent Motion Diffusion for Video Generation](https://arxiv.org/abs/2304.11603) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.11603) | - | - | Apr., 2023 |
|[Conditional Image-to-Video Generation with Latent Flow Diffusion Models](https://arxiv.org/abs/2303.13744) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.13744) | [![Star](https://img.shields.io/github/stars/nihaomiao/CVPR23_LFDM.svg?style=social&label=Star)](https://github.com/nihaomiao/CVPR23_LFDM) | - | CVPR 2023 |


### Brain-guided Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity](https://arxiv.org/abs/2305.11675) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11675) | [![Star](https://img.shields.io/github/stars/jqin4749/MindVideo.svg?style=social&label=Star)](https://github.com/jqin4749/MindVideo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://mind-video.com/) | May, 2023 |


## Depth-guided Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation](https://arxiv.org/abs/2307.06940) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.06940) | [![Star](https://img.shields.io/github/stars/VideoCrafter/Animate-A-Story.svg?style=social&label=Star)](https://github.com/VideoCrafter/Animate-A-Story) | [![Website](https://img.shields.io/badge/Website-9cf)](https://videocrafter.github.io/Animate-A-Story/) | Jul., 2023 |
|[Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance](https://arxiv.org/abs/2306.00943) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.00943) | [![Star](https://img.shields.io/github/stars/VideoCrafter/Make-Your-Video.svg?style=social&label=Star)](https://github.com/VideoCrafter/Make-Your-Video) | [![Website](https://img.shields.io/badge/Website-9cf)](https://doubiiu.github.io/projects/Make-Your-Video/) | Jun., 2023 |



### Multi-modal guided Video Generation

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[VideoComposer: Compositional Video Synthesis with Motion Controllability](https://arxiv.org/abs/2306.02018) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02018) | [![Star](https://img.shields.io/github/stars/damo-vilab/videocomposer.svg?style=social&label=Star)](https://github.com/damo-vilab/videocomposer) | [![Website](https://img.shields.io/badge/Website-9cf)](https://videocomposer.github.io/) | Jun., 2023 |
|[NExT-GPT: Any-to-Any Multimodal LLM](https://arxiv.org/abs/2309.05519) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.05519) | - | - | Sep, 2023 |
|[MovieFactory: Automatic Movie Creation from Text using Large Generative Models for Language and Images](https://arxiv.org/pdf/2306.07257.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2306.07257.pdf) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://www.bilibili.com/video/BV1qj411Q76P/) | Jun, 2023 |
|[Any-to-Any Generation via Composable Diffusion](https://arxiv.org/abs/2305.11846) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.11846) | [![Star](https://img.shields.io/github/stars/microsoft/i-Code.svg?style=social&label=Star)](https://github.com/microsoft/i-Code/tree/main/i-Code-V3) | [![Website](https://img.shields.io/badge/Website-9cf)](https://codi-gen.github.io/) | May, 2023 |
|[Mm-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation](https://arxiv.org/abs/2212.09478) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.09478) | [![Star](https://img.shields.io/github/stars/researchmm/MM-Diffusion.svg?style=social&label=Star)](https://github.com/researchmm/MM-Diffusion) | - | CVPR 2023 |


## Unconditional Video Generation

### U-Net based


| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[Video Probabilistic Diffusion Models in Projected Latent Space](https://arxiv.org/abs/2302.07685) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.07685) | [![Star](https://img.shields.io/github/stars/sihyun-yu/PVDM.svg?style=social&label=Star)](https://github.com/sihyun-yu/PVDM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://sihyun.me/PVDM/) | CVPR 2023 |
|[VIDM: Video Implicit Diffusion Models](https://arxiv.org/abs/2212.00235) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.00235) | [![Star](https://img.shields.io/github/stars/MKFMIKU/VIDM.svg?style=social&label=Star)](https://github.com/MKFMIKU/VIDM) | [![Website](https://img.shields.io/badge/Website-9cf)](https://kfmei.page/vidm/) | AAAI 2023 |
|[GD-VDM: Generated Depth for better Diffusion-based Video Generation](https://arxiv.org/abs/2306.11173) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.11173) | [![Star](https://img.shields.io/github/stars/lapid92/GD-VDM.svg?style=social&label=Star)](https://github.com/lapid92/GD-VDM) | - | Jun., 2023 |
|[LEO: Generative Latent Image Animator for Human Video Synthesis](https://arxiv.org/pdf/2305.03989.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.03989.pdf) | [![Star](https://img.shields.io/github/stars/wyhsirius/LEO.svg?style=social&label=Star)](https://github.com/wyhsirius/LEO) | [![Website](https://img.shields.io/badge/Website-9cf)](https://wyhsirius.github.io/LEO-project/) | May., 2023 |

### Transformer based

| Title | arXiv | Github | WebSite | Pub. & Date |
|---|---|---|---|---|
|[VDT: An Empirical Study on Video Diffusion with Transformers](https://arxiv.org/abs/2305.13311) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13311) | [![Star](https://img.shields.io/github/stars/RERV/VDT.svg?style=social&label=Star)](https://github.com/RERV/VDT) | - | May, 2023 |


## Video Completion

### Video Enhancement and Restoration


| Title | arXiv | Github | Pub. & Date |
|---|---|---|---|
|[LDMVFI: Video Frame Interpolation with Latent Diffusion Models](https://arxiv.org/abs/2303.09508) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09508) | - | Mar., 2023 |
|[CaDM: Codec-aware Diffusion Modeling for Neural-enhanced Video Streaming](https://arxiv.org/abs/2211.08428) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.08428) | - | Nov., 2022 |
| [Look Ma, No Hands! Agent-Environment Factorization of Egocentric Videos](https://arxiv.org/pdf/2305.16301.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.16301.pdf) | - | - | May., 2023 |






### Video Prediction

| Title | arXiv | Github | Website | Pub. & Date |
|---|---|---|---|---|
|[Video Diffusion Models with Local-Global Context Guidance](https://arxiv.org/pdf/2306.02562.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2306.02562.pdf) | [![Star](https://img.shields.io/github/stars/exisas/LGC-VD.svg?style=social&label=Star)](https://github.com/exisas/LGC-VD) | - | IJCAI, 2023 |
|[Seer: Language Instructed Video Prediction with Latent Diffusion Models](https://arxiv.org/abs/2303.14897) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14897) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://seervideodiffusion.github.io/) | Mar., 2023 |
|[Diffusion Models for Video Prediction and Infilling](https://arxiv.org/abs/2206.07696) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2206.07696) | [![Star](https://img.shields.io/github/stars/Tobi-r9/RaMViD.svg?style=social&label=Star)](https://github.com/Tobi-r9/RaMViD) | [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/video-diffusion-prediction) | TMLR 2022 |
|[McVd: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation](https://arxiv.org/abs/2205.09853) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.09853) | [![Star](https://img.shields.io/github/stars/Tobi-r9/RaMViD.svg?style=social&label=Star)](https://github.com/voletiv/mcvd-pytorch) | [![Website](https://img.shields.io/badge/Website-9cf)](https://mask-cond-video-diffusion.github.io) | NeurIPS 2022 |
|[Diffusion Probabilistic Modeling for Video Generation](https://arxiv.org/abs/2203.09481) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2203.09481) | [![Star](https://img.shields.io/github/stars/buggyyang/RVD.svg?style=social&label=Star)](https://github.com/buggyyang/RVD) | - | Mar., 2022 |
|[Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2205.11495) | [![Star](https://img.shields.io/github/stars/plai-group/flexible-video-diffusion-modeling.svg?style=social&label=Star)](https://github.com/plai-group/flexible-video-diffusion-modeling) | [![Website](https://img.shields.io/badge/Website-9cf)](https://fdmolv.github.io/) | May, 2022 |
|[Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models](https://arxiv.org/abs/2305.13840) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13840) | [![Star](https://img.shields.io/github/stars/Weifeng-Chen/control-a-video.svg?style=social&label=Star)](https://github.com/Weifeng-Chen/control-a-video) | [![Website](https://img.shields.io/badge/Website-9cf)](https://controlavideo.github.io/) | May, 2023 |




## Video Editing 

### General Editing Model


| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation](https://arxiv.org/abs/2309.00908) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.00908) | - | - | Sep, 2023 |
| [MagicEdit: High-Fidelity and Temporally Coherent Video Editing](https://arxiv.org/abs/2308.14749) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.14749) | - | - | Aug, 2023 |
| [Edit Temporal-Consistent Videos with Image Diffusion Model](https://arxiv.org/abs/2308.09091) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.09091) | - | - | Aug, 2023 |
| [Structure and Content-Guided Video Synthesis With Diffusion Models](https://arxiv.org/abs/2302.03011) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.03011) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://research.runwayml.com/gen2) | ICCV, 2023 |
| [Dreamix: Video Diffusion Models Are General Video Editors](https://arxiv.org/abs/2302.01329) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2302.01329) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://dreamix-video-editing.github.io/) | Feb, 2023 |



### Training-free Editing Model

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [LOVECon: Text-driven Training-Free Long Video Editing with ControlNet](https://arxiv.org/abs/2310.09711)| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.09711) | [![Star](https://img.shields.io/github/stars/zhijie-group/LOVECon.svg?style=social&label=Star)](https://github.com/zhijie-group/LOVECon)  | - | Oct., 2023 |
| [FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing](https://arxiv.org/abs/2310.05922) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.05922) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://flatten-video-editing.github.io/) | Oct., 2023 |
| [Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models](https://arxiv.org/abs/2310.01107) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.01107) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://ground-a-video.github.io/) | Oct., 2023 |
| [MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance](https://arxiv.org/pdf/2308.10079.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.10079.pdf) | - | - | Aug., 2023 |
| [EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints](https://arxiv.org/pdf/2308.10648.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.10648.pdf) | - | - | Aug., 2023 |
| [ControlVideo: Training-free Controllable Text-to-Video Generation](https://arxiv.org/abs/2305.13077) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13077) | [![Star](https://img.shields.io/github/stars/YBYBZhang/ControlVideo.svg?style=social&label=Star)](https://github.com/YBYBZhang/ControlVideo) | - | May, 2023 |
| [TokenFlow: Consistent Diffusion Features for Consistent Video Editing](https://arxiv.org/abs/2307.10373) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.10373) | [![Star](https://img.shields.io/github/stars/omerbt/TokenFlow.svg?style=social&label=Star)](https://github.com/omerbt/TokenFlow) | [![Website](https://img.shields.io/badge/Website-9cf)](https://diffusion-tokenflow.github.io/) | Jul., 2023 |
| [VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing](https://arxiv.org/abs//2306.08707) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs//2306.08707) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://videdit.github.io/) | Jun., 2023 |
| [Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation](https://arxiv.org/abs/2306.07954) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.07954) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://anonymous-31415926.github.io/) | Jun., 2023 |
| [Zero-Shot Video Editing Using Off-the-Shelf Image Diffusion Models](https://arxiv.org/abs/2303.17599) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.17599) | [![Star](https://img.shields.io/github/stars/baaivision/vid2vid-zero.svg?style=social&label=Star)](https://github.com/baaivision/vid2vid-zero) | [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/spaces/BAAI/vid2vid-zero) | Mar., 2023 |
| [FateZero: Fusing Attentions for Zero-shot Text-based Video Editing](https://arxiv.org/abs/2303.09535) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09535) | [![Star](https://img.shields.io/github/stars/ChenyangQiQi/FateZero.svg?style=social&label=Star)](https://github.com/ChenyangQiQi/FateZero) | [![Website](https://img.shields.io/badge/Website-9cf)](https://fate-zero-edit.github.io/) | Mar., 2023 |
| [Pix2video: Video Editing Using Image Diffusion](https://arxiv.org/abs/2303.12688) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.12688) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://duyguceylan.github.io/pix2video.github.io/) | Mar., 2023 |
| [InFusion: Inject and Attention Fusion for Multi Concept Zero Shot Text based Video Editing](https://arxiv.org/abs/2308.00135) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.00135) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://infusion-zero-edit.github.io/) | Aug., 2023 |
|[Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising](https://arxiv.org/abs/2305.18264) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.18264) | [![Star](https://img.shields.io/github/stars/G-U-N/Gen-L-Video.svg?style=social&label=Star)](https://github.com/G-U-N/Gen-L-Video) | [![Website](https://img.shields.io/badge/Website-9cf)](https://g-u-n.github.io/projects/gen-long-video/index.html) | May, 2023 |


### One-shot Editing Model

| Title | arXiv | Github | Website | Pub. & Date |
|---|---|---|---|---|
| [StableVideo: Text-driven Consistency-aware Diffusion Video Editing](https://arxiv.org/abs/2308.09592) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.09592) | [![Star](https://img.shields.io/github/stars/rese1f/StableVideo?style=social)](https://github.com/rese1f/StableVideo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://rese1f.github.io/StableVideo) | ICCV, 2023 |
| [Shape-aware Text-driven Layered Video Editing](https://arxiv.org/pdf/2301.13173.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2301.13173.pdf) | - | - | CVPR, 2023 |
| [SAVE: Spectral-Shift-Aware Adaptation of Image Diffusion Models for Text-guided Video Editing](https://arxiv.org/pdf/2305.18670.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.18670.pdf) | [![Star](https://img.shields.io/github/stars/nazmul-karim170/SAVE-Text2Video?style=social)](https://github.com/nazmul-karim170/SAVE-Text2Video) | - | May, 2023 |
| [Towards Consistent Video Editing with Text-to-Image Diffusion Models](https://arxiv.org/pdf/2305.17431.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.17431.pdf) | - | - | Mar., 2023 |
| [Edit-A-Video: Single Video Editing with Object-Aware Consistency](https://arxiv.org/abs/2303.07945) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.07945) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://edit-a-video.github.io/) | Mar., 2023 |
| [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.11565) | [![Star](https://img.shields.io/github/stars/showlab/Tune-A-Video?style=social)](https://github.com/showlab/Tune-A-Video) | [![Website](https://img.shields.io/badge/Website-9cf)](https://tuneavideo.github.io/) | ICCV, 2023 |
| [ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing](https://arxiv.org/abs/2305.17098) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.17098) | [![Star](https://img.shields.io/github/stars/thu-ml/controlvideo.svg?style=social&label=Star)](https://github.com/thu-ml/controlvideo) | [![Website](https://img.shields.io/badge/Website-9cf)](https://ml.cs.tsinghua.edu.cn/controlvideo/) | May, 2023 |
| [Video-P2P: Video Editing with Cross-attention Control](https://arxiv.org/abs/2303.04761) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.04761) | [![Star](https://img.shields.io/github/stars/ShaoTengLiu/Video-P2P.svg?style=social&label=Star)](https://github.com/ShaoTengLiu/Video-P2P) | [![Website](https://img.shields.io/badge/Website-9cf)](https://video-p2p.github.io/) | Mar., 2023 |
|[SinFusion: Training Diffusion Models on a Single Image or Video](https://arxiv.org/abs/2211.11743) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.11743) | [![Star](https://img.shields.io/github/stars/YingqingHe/LVDM.svg?style=social&label=Star)](https://github.com/yanivnik/sinfusion-code) | [![Website](https://img.shields.io/badge/Website-9cf)](https://yanivnik.github.io/sinfusion/) | Nov., 2022 |









### Instruct-guided Video Editing

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [InstructVid2Vid: Controllable Video Editing with Natural Language Instructions](https://arxiv.org/pdf/2305.12328.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.12328.pdf) | - | - | May, 2023 |
| [Collaborative Score Distillation for Consistent Visual Synthesis](https://arxiv.org/pdf/2307.04787.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2307.04787.pdf) | - | - | July, 2023 |

### Motion-guided Video Editing
| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet](https://arxiv.org/pdf/2307.14073.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2307.14073.pdf) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://vcg-aigc.github.io/) | July, 2023 |


### Sound-guided Video Editing

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [Speech Driven Video Editing via an Audio-Conditioned Diffusion Model](https://arxiv.org/pdf/2301.04474.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2301.04474.pdf) | - | - | May., 2023 |
| [Soundini: Sound-Guided Diffusion for Natural Video Editing](https://arxiv.org/abs/2304.06818) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.06818) | [![Star](https://img.shields.io/github/stars/kuai-lab/soundini-official.svg?style=social&label=Star)](https://github.com/kuai-lab/soundini-official) | [![Website](https://img.shields.io/badge/Website-9cf)](https://kuai-lab.github.io/soundini-gallery/) | Apr., 2023 |


### Multi-modal Control Editing Model

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [CCEdit: Creative and Controllable Video Editing via Diffusion Models](https://arxiv.org/abs/2309.16496) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2309.16496) | - | - | Sep, 2023 |
| [Make-A-Protagonist: Generic Video Editing with An Ensemble of Experts](https://arxiv.org/abs/2305.08850) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.08850) | [![Star](https://img.shields.io/github/stars/Make-A-Protagonist/Make-A-Protagonist.svg?style=social&label=Star)](https://github.com/Make-A-Protagonist/Make-A-Protagonist) | [![Website](https://img.shields.io/badge/Website-9cf)](https://make-a-protagonist.github.io/) | May, 2023 |




### Domain-specific Editing Model

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [Multimodal-driven Talking Face Generation via a Unified Diffusion-based Generator](https://arxiv.org/pdf/2305.02594.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.02594.pdf) | - | - | May, 2023 |
| [DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis](https://arxiv.org/pdf/2308.03463.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.03463.pdf) | - | - | Aug, 2023 |
| [Style-A-Video: Agile Diffusion for Arbitrary Text-based Video Style Transfer](https://arxiv.org/abs/2305.05464) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.05464) | [![Star](https://img.shields.io/github/stars/haha-lisa/Style-A-Video.svg?style=social&label=Star)](https://github.com/haha-lisa/Style-A-Video) | - | May, 2023 |
| [Instruct-Video2Avatar: Video-to-Avatar Generation with Instructions](https://arxiv.org/abs/2306.02903) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.02903) | [![Star](https://img.shields.io/github/stars/lsx0101/Instruct-Video2Avatar.svg?style=social&label=Star)](https://github.com/lsx0101/Instruct-Video2Avatar) | - | Jun, 2023 |
| [Video Colorization with Pre-trained Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.01732) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2306.01732) | [![Star](https://img.shields.io/github/stars/ColorDiffuser/ColorDiffuser.svg?style=social&label=Star)](https://github.com/ColorDiffuser/ColorDiffuser) | [![Website](https://img.shields.io/badge/Website-9cf)](https://colordiffuser.github.io/) | Jun, 2023 |
| [Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding](https://arxiv.org/abs/2212.02802) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.02802) | [![Star](https://img.shields.io/github/stars/man805/Diffusion-Video-Autoencoders.svg?style=social&label=Star)](https://github.com/man805/Diffusion-Video-Autoencoders) | [![Website](https://img.shields.io/badge/Website-9cf)](https://diff-video-ae.github.io/) | CVPR 2023 |


### Non-diffusion Editing model

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing](https://arxiv.org/abs/2310.10624)| [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2310.10624) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://showlab.github.io/DynVideo-E/) | Oct., 2023 |
| [INVE: Interactive Neural Video Editing](https://arxiv.org/abs/2307.07663) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.07663) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://gabriel-huang.github.io/inve/) | Jul., 2023 |
| [Shape-Aware Text-Driven Layered Video Editing](https://arxiv.org/abs/2301.13173) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2301.13173) | - | [![Website](https://img.shields.io/badge/Website-9cf)](https://text-video-edit.github.io/) | Jan., 2023 |




### Video Understanding

| Title | arXiv | Github | Website | Pub. Date |
|---|---|---|---|---|
| [DiffusionVMR: Diffusion Model for Video Moment Retrieval](https://arxiv.org/abs/2308.15109) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2308.15109) | - | - | Aug., 2023 |
| [DiffPose: SpatioTemporal Diffusion Model for Video-Based Human Pose Estimation](https://arxiv.org/pdf/2307.16687.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2307.16687.pdf) | - | - | Aug., 2023 |
| [Unsupervised Video Anomaly Detection with Diffusion Models Conditioned on Compact Motion Representations](https://arxiv.org/abs/2307.01533) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.01533) | - | - | ICIAP, 2023 |
| [Exploring Diffusion Models for Unsupervised Video Anomaly Detection](https://arxiv.org/abs/2304.05841) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2304.05841) | - | - | Apr., 2023 |
| [Multimodal Motion Conditioned Diffusion Model for Skeleton-based Video Anomaly Detection](https://arxiv.org/abs/2307.07205) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.07205) | - | - | ICCV, 2023 |
| [Diffusion Action Segmentation](https://arxiv.org/pdf/2303.17959.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2303.17959.pdf) | - | - | Mar., 2023 |
| [DiffTAD: Temporal Action Detection with Proposal Denoising Diffusion](https://arxiv.org/abs/2303.14863) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14863) |[![Star](https://img.shields.io/github/stars/sauradip/DiffusionTAD.svg?style=social&label=Star)](https://github.com/sauradip/DiffusionTAD)  | [![Website](https://img.shields.io/badge/Website-9cf)](https://arxiv.org/abs/2303.14863) | Mar., 2023 |
| [DiffusionRet: Generative Text-Video Retrieval with Diffusion Model](https://arxiv.org/abs/2303.09867) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.09867) | - | - | Mar., 2023 |
| [MomentDiff: Generative Video Moment Retrieval from Random to Real](https://arxiv.org/pdf/2307.02869.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2307.02869.pdf) |[![Star](https://img.shields.io/github/stars/IMCCretrieval/MomentDiff.svg?style=social&label=Star)](https://github.com/IMCCretrieval/MomentDiff) | [![Website](https://img.shields.io/badge/Website-9cf)](https://arxiv.org/pdf/2307.02869.pdf) | Jul., 2023 |
| [Refined Semantic Enhancement Towards Frequency Diffusion for Video Captioning](https://arxiv.org/abs/2211.15076) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2211.15076) | - | - | Nov., 2022 |
| [A Generalist Framework for Panoptic Segmentation of Images and Videos](https://arxiv.org/abs/2210.06366) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2210.06366) | [![Star](https://img.shields.io/github/stars/google-research/pix2seq.svg?style=social&label=Star)](https://github.com/google-research/pix2seq) | [![Website](https://img.shields.io/badge/Website-9cf)](https://arxiv.org/abs/2210.06366) | Oct., 2022 |
| [DAVIS: High-Quality Audio-Visual Separation with Generative Diffusion Models](https://arxiv.org/pdf/2308.00122.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2308.00122.pdf) | - | - | Jul., 2023 |
| [CaDM: Codec-aware Diffusion Modeling for Neural-enhanced Video Streaming](https://arxiv.org/pdf/2211.08428.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2211.08428.pdf) | - | - | Mar., 2023 |
| [Spatial-temporal Transformer-guided Diffusion based Data Augmentation for Efficient Skeleton-based Action Recognition](https://arxiv.org/pdf/2302.13434.pdf) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2302.13434.pdf) | - | - | Jul., 2023 |
|[PDPP: Projected Diffusion for Procedure Planning in Instructional Videos](https://arxiv.org/abs/2303.14676) | [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2303.14676) | [![Star](https://img.shields.io/github/stars/MCG-NJU/PDPP.svg?style=social&label=Star)](https://github.com/MCG-NJU/PDPP) | - |CVPR 2023 |

## Contact
If you have any suggestions or find our work helpful, feel free to contact us

Homepage: [Zhen Xing](https://chenhsing.github.io)

Email: zhenxingfd@gmail.com


If you find our work useful, please consider citing it:

```
@article{vdmsurvey,
  title={A Survey on Video Diffusion Models},
  author={Zhen Xing and Qijun Feng and Haoran Chen and Qi Dai and Han Hu and Hang Xu and Zuxuan Wu and Yu-Gang Jiang}, 
  journal={arXiv preprint arXiv:2310.10647},
  year={2023}
}
```

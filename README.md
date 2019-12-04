# Audio-Visual Scene-Aware Dialog

code for the paper:
**[AVSD][1]**
Huda Alamri, Vincent Cartillier, Abhishek Das, Jue Wang, Stefan Lee, Peter Anderson, Irfan Essa, Devi Parikh, Dhruv Batra, Anoop Cherian, Tim K. Marks, Chiori Hori


website:
[video-dialog.com][3]

This code has been developed upon [batra-mlp-lab/visdial-challenge-starter-pytorch][2]

## Setup
```sh
# create and activate environment
conda env create -n avsd -f=env.yml
conda activate avsd
```

## Data
 * download 'split'.json data at: video-dialog.com
 * Extracted video, audio, and dialog features can be downloaded from 
 [here](https://drive.google.com/drive/folders/14zlHmNFkCgptiGttwWKrsaaz5vVUFs00?usp=sharing)


## Workflow

 * Build dialogs json file with otions using ```makejson_with_options.py``` (output: 'split'_options.json)
 * Adapt JSON format using ```convert_json_to_visdial_style.py``` (output: 'split'_options_2.json can be renamed after to 'split'_options.json)
 * Build tokenized captions, dialogs and image paths with ```prepro.py``` (output: dialogs.h5 and params.json)
 * Build the image features (if working with images) using ```prepro_img_vgg16.lua``` or ```prepro_img_resnet.lua```  from the [batra-mlp-lab/visdial-challenge-starter-pytorch][2] (output: data_img.h5)
 * Build video features I3D (output: data_video.h5) [https://github.com/piergiaj/pytorch-i3d.git][5]
 * Build audio features AENET (output: data_audio.h5) [https://github.com/znaoya/aenet.git][4]


 * Training: python train.py
 * evaluation: python evaluate.py --use_gt

## If you find this code useful in your research, please consider citing:
```
@article{DBLP:journals/corr/abs-1901-09107,
  author    = {Huda Alamri and
               Vincent Cartillier and
               Abhishek Das and
               Jue Wang and
               Stefan Lee and
               Peter Anderson and
               Irfan Essa and
               Devi Parikh and
               Dhruv Batra and
               Anoop Cherian and
               Tim K. Marks and
               Chiori Hori},
  title     = {Audio-Visual Scene-Aware Dialog},
  journal   = {CoRR},
  volume    = {abs/1901.09107},
  year      = {2019},
  url       = {http://arxiv.org/abs/1901.09107},
  archivePrefix = {arXiv},
  eprint    = {1901.09107},
  timestamp = {Sat, 02 Feb 2019 16:56:00 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1901-09107},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
BSD

[1]: https://arxiv.org/abs/1806.00525
[2]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[3]: https://video-dialog.com
[4]: https://github.com/znaoya/aenet.git
[5]: https://github.com/piergiaj/pytorch-i3d.git

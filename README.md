[keras] Dataset Augmentation in Feature Space
===
Tensorflow implementation is provided as the following link.  
https://github.com/YeongHyeon/Sequence-Autoencoder

## Architecture
### Encoder
<img src="./figures/encoder_model.png" width="100">  

### Decoder
<img src="./figures/decoder_model.png" width="200">  

## Problem Definition
Recunstruct pen sequence.  
<img src="./figures/autoencoder_structure.png" width="400"> 

## Result
- Result of simple reconstruction  
<img src="./figures/gen_8.png" width="150"> 
<img src="./figures/gen_w.png" width="150"> 
<img src="./figures/gen_k.png" width="150"> 
- Result of latent space interpolation and extrapolation  
<img src="./figures/latent_augmentation.png" width="200"> 

## Environment
- Python : 3.7.10  
- Tensorflow ; 2.4.1  
- keras : 2.4.3  
- Numpy : 1.19.5  

## Reference
[1] Terrance DeVries, Graham W. Taylor. (2017). <a href="https://arxiv.org/abs/1702.05538">Dataset Augmentation in Feature Space</a>. arXiv preprint arXiv:1702.05538.  
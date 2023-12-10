# Bimodal SegNet: Instance Segmentation Fusing Events and RGB Frames for Robotic Grasping

# Cite the article:
@misc{kachole2023bimodal, 
      title={Bimodal SegNet: Instance Segmentation Fusing Events and RGB Frames for Robotic Grasping}, 
      author={Sanket Kachole and Xiaoqian Huang and Fariborz Baghaei Naeini and Rajkumar Muthusamy and Dimitrios Makris and Yahya Zweiri},
      year={2023},
      eprint={2303.11228},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

# Dataset repository:
https://kuacae-my.sharepoint.com/personal/100049863_ku_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F100049863%5Fku%5Fac%5Fae%2FDocuments%2FESD&ga=1


# Framework of Bimodal SegNet

Object segmentation for robotic grasping under dynamic conditions often faces challenges such as occlusion, low light conditions, motion blur and object size variance. To address these challenges, we propose a Deep Learning network that fuses two types of visual signals, event-based data and RGB  frame data. The proposed Bimodal SegNet network has two distinct encoders,  one for each signal input and a spatial pyramidal pooling with atrous convolutions. Encoders capture rich contextual information by pooling the concatenated features at different resolutions while the decoder obtains sharp object boundaries. The evaluation of the proposed method undertakes five unique image degradation challenges including occlusion, blur, brightness, trajectory and scale variance on the Event-based Segmentation (ESD) Dataset. The evaluation results show a 6-10\% segmentation accuracy improvement over state-of-the-art methods in terms of mean intersection over the union and pixel accuracy.

 <img width="516" alt="Overview Architecture" src="https://github.com/sanket0707/Bimodal-SegNet/assets/43345233/4337f390-d334-4b7c-9ec8-7a2f37b9ca1b">

# Code Implementation

# Requirements:
#####  Python 3.7 
##### Tensorflow 2.11.0








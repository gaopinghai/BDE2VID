# Enhancing Event-based Video Reconstruction with Bidirectional Temporal Information
This is the code for **Enhancing Event-based Video Reconstruction with Bidirectional Temporal Information** by Pinghai Gao, Longguang Wang, Sheng Ao, Ye Zhang and Yulan Guo:

![media/cmp_res.png](https://github.com/gaopinghai/BDE2VID/blob/main/media/cmp_res.png)

[![temporal_receptive_field_comparision](https://res.cloudinary.com/marcomontalbano/image/upload/v1692347380/video_to_markdown/images/youtube--Tfww130Meks-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://youtu.be/Tfww130Meks "temporal_receptive_field_comparision")

## Datasets
The HQF dataset can be downloaded from this [link](https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_?usp=sharing).

## Pretrained Weights
The pretrained model, which can reproduce the quantitative results in the paper, can be downloaded from this [site](https://drive.google.com/file/d/1mHa_zf5icxf_Mm2JFnfI7KN76oRTDqef/view?usp=sharing).

## Inference
To run the evaluation code, you can use command:
```python
python eval_models.py --dir weights/
```
This command will evaluate all the `.pth` files in the `weights/` dir.

## Related Projects
- [High Speed and High Dynamic Range Video with an Event Camera](https://github.com/uzh-rpg/rpg_e2vid)
- [Reducing the Sim-to-Real Gap for Event Cameras](https://timostoff.github.io/20ecnn)
- [Event-based Video Reconstruction Using Transformer](https://github.com/WarranWeng/ET-Net)
- [Spatially-Adaptive Denormalization for Event-Based Video Reconstruction](https://github.com/RodrigoGantier/SPADE_E2VID)
- [Event-based Video Reconstruction via Potential-assisted Spiking Neural Network](https://sites.google.com/view/evsnn)
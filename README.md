# Enhancing Event-based Video Reconstruction with Bidirectional Temporal Information
This is the code for **Enhancing Event-based Video Reconstruction with Bidirectional Temporal Information** by Pinghai Gao, Longguang Wang, Sheng Ao, Ye Zhang and Yulan Guo.

## Datasets
The datasets for evaluation can be downloaded from this [link](https://huggingface.co/datasets/pinghai/BDE2VID_Datasets). Download them to `data/eval/h5/` for default settings.

## Pretrained Weights
The pretrained model, `BDE2VID.pth`, which can reproduce the quantitative results in the paper, is in the `weights` directory.

## Inference
To run the evaluation code, you can use command:
```python
python eval_models_seq.py --weights_dir weights/ --data_dir data/eval/h5
```
This command will evaluate all the `.pth` files in the `weights/` dir.

## Related Projects
- [High Speed and High Dynamic Range Video with an Event Camera](https://github.com/uzh-rpg/rpg_e2vid)
- [Reducing the Sim-to-Real Gap for Event Cameras](https://timostoff.github.io/20ecnn)
- [Event-based Video Reconstruction Using Transformer](https://github.com/WarranWeng/ET-Net)
- [Spatially-Adaptive Denormalization for Event-Based Video Reconstruction](https://github.com/RodrigoGantier/SPADE_E2VID)
- [Event-based Video Reconstruction via Potential-assisted Spiking Neural Network](https://sites.google.com/view/evsnn)
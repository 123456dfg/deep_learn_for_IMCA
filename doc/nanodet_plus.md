# 关于使用Nanodet-Plus训练模型（必须使用Cuda加速）


1. 安装CUDA和Cudnn
2. 安装pytorch-cuda版
3. 克隆Nanodet-plus

* <font color="Red">*不要修改顺序!!!*</font>

```SH
# 最好是在Conda或venv环境下运行
git clone https://github.com/RangiLyu/nanodet.git
cd nanodet
pip install -r requirements.txt
python setup.py develop
把这里的nanodet文件夹中的tools替换掉clone下来的
```
4.安装cudatoolkit
```
pip install cudatoolkit=11.2 -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

5. 制作VOC格式的数据集，并保证你的文件目录如下图所视

```
VOC_Dataset
    ├───Annotations
    │       ├─── **.xml
    └───JPEGImages
            └─── **.jpg
```

6. 生成label标签，~~工具在本项目的tools/generate_label.py（不是nanodet里的tools）~~（现已整合到nanodet的tools中），最好生成在如上面所视的VOC_Dataset文件夹里面
7. 转换成coco格式，~~转换工具在本项目的tools/voc2coco.py（不是nanodet里的tools）~~（现已整合到nanodet的tools中）

```SH
# 执行下面的命令，注意要先修改参数
python tools/x2coco.py \
--dataset_type voc \
--image_input_dir JPEGImages \
--voc_anno_dir Annotations \
--voc_anno_list test.txt \
--voc_label_list label_list.txt \
--output_dir coco \
--voc_out_name 'test.json'
```

8. 修改config/nanodet-plus-m_416.yml中的配置参数

```YAML
# nanodet-plus-m_416
# COCO mAP(0.5:0.95) = 0.304
#             AP_50  = 0.459
#             AP_75  = 0.317
#           AP_small = 0.106
#               AP_m = 0.322
#               AP_l = 0.477
save_dir: workspace/nanodet-plus-leaf-416 # 训练完的模型存放位置
model:
  weight_averager:
    name: ExpMovingAverager
    decay: 0.9998
  arch:
    name: NanoDetPlus
    detach_epoch: 10
    backbone:
      name: ShuffleNetV2
      model_size: 1.0x
      out_stages: [2,3,4]
      activation: LeakyReLU
    fpn:
      name: GhostPAN
      in_channels: [116, 232, 464]
      out_channels: 96
      kernel_size: 5
      num_extra_level: 1
      use_depthwise: True
      activation: LeakyReLU
    head:
      name: NanoDetPlusHead
      num_classes: 1
      input_channel: 96
      feat_channels: 96
      stacked_convs: 2
      kernel_size: 5
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7
      norm_cfg:
        type: BN
      loss:
        loss_qfl:
          name: QualityFocalLoss
          use_sigmoid: True
          beta: 2.0
          loss_weight: 1.0
        loss_dfl:
          name: DistributionFocalLoss
          loss_weight: 0.25
        loss_bbox:
          name: GIoULoss
          loss_weight: 2.0
    # Auxiliary head, only use in training time.
    aux_head:
      name: SimpleConvHead
      num_classes: 1
      input_channel: 192
      feat_channels: 192
      stacked_convs: 4
      strides: [8, 16, 32, 64]
      activation: LeakyReLU
      reg_max: 7
data:
  train:
    name: CocoDataset
    img_path: # 图片路径
    ann_path: # 训练集json文件
    input_size: [416,416] #[w,h]
    keep_ratio: False
    pipeline:
      perspective: 0.0
      scale: [0.6, 1.4]
      stretch: [[0.8, 1.2], [0.8, 1.2]]
      rotation: 0
      shear: 0
      translate: 0.2
      flip: 0.5
      brightness: 0.2
      contrast: [0.6, 1.4]
      saturation: [0.5, 1.2]
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]
  val:
    name: CocoDataset
    img_path: # 图片路径
    ann_path: # 验证集json文件
    input_size: [416,416] #[w,h]
    keep_ratio: False
    pipeline:
      normalize: [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]] # 有空再管这个
device:
  gpu_ids: [0]
  workers_per_gpu: 10
  batchsize_per_gpu: 16 # 每一批的训练个数
  precision: 32 # set to 16 to use AMP training
schedule:
#  resume:
#  load_model:
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.05
  warmup:
    name: linear
    steps: 500
    ratio: 0.0001
  total_epochs: 300
  lr_schedule:
    name: CosineAnnealingLR
    T_max: 300
    eta_min: 0.00005
  val_intervals: 10
grad_clip: 35
evaluator:
  name: CocoDetectionEvaluator
  save_key: mAP
log:
  interval: 50

class_names: ['leaf'] ## 标签
```

9. 运行

```SH
python tools/train.py config/nanodet-plus-m_416.yml
```

10. 将pth转换成onnx类型

```SH
python ./tools/export_onnx.py --cfg_path ./config/nanodet-plus-m_416.yml(改成你的yml路径) --model_path ./workspace/nanodet-plus-leaf-416/model_best/nanodet_model_best.pth(改成你的pth路径) --out_path ./workspace/nanodet-plus-leaf-416/model_best/leaf.onnx(改成你要输出的onnx路径)
```

11.将onnx转换成xml和bin类型
```SH
mo --input_model /home/dfg/leaf.onnx(改成你的onnx路径) --mean_values data[103.53,116.28,123.675](改成你的平均标量) --scale_values data[57.375,57.12,58.395](改成你的平均标量) --output output --output_dir /home/dfg(改成你要输出的路径)
```
## 环境配置
### python环境
与[EDVR](https://github.com/xinntao/EDVR)相同。其中比较关键的是torch用1.1.0

`pip install pillow==6.1 torch==1.1.0 torchvision==0.3.0`
### 准备dcn

```
cd codes/dcn
python setup.py develop
```

### 安装ffmpeg
教程：https://baijiahao.baidu.com/s?id=1660327134602942057&wfr=spider&for=pc


## 准备数据集

### 安装ffmpeg-python工具包

`pip install ffmpeg-python`
### 生成退化数据集
    
`python gray.py --p 0.1 0.3 0.6 --input 视频路径 --Scratches 划痕素材路径 --output 输出路径`

### （可选）生成lmdb文件
lmdb格式文件理论上可提高训练时数据的读取速度。

`python create_lmdb.py --dataset ../datasets/Validation --output ../datasets/Validation --name Validation`

## 训练

示例

`python main.py --save_folder ../experiments/experiment1 --train_dir /content/drive/Shared\ drives/furu.zhao/VFNet/datasets/Test2 --batchSize 2 --nFrames 5 --groups 8 --val_dir /content/drive/Shared\ drives/furu.zhao/VFNet/datasets/Validation --shuffle --pretrained True --pretrained_sr epoch_8.pth`

## 测试
（之后再补）

`python test.py ..........`

#### ....to be continued
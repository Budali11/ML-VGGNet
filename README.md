# ML-VGGNet
复现VGGNet16
使用ILSVRC2012数据集，训练集和验证集都需要分类到类似`nXXXXXXX`命名的子文件夹中。
训练VGG11：
```
python train_vgg11_gemini.py
```
完成后会生成权重文件，再训练VGG16（需要指定VGG11权重文件路径）：
```
python train_vgg16_from_vgg11.py
```
测试：
```
python test_vgg16_gemini.py
```
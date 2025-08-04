# UNet-Pytorch 代码精简版

模型的构建部分参考了github开源项目 annotated_deep_learning_paper_implementations

[https://github.com/labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

train.py 为训练的代码(使用了Tensorboard来记录训练过程)

pred.py 为推理的代码

在ISTD测试集的结果为(反归一化后计算):

```
Average PSNR: 27.2417
Average SSIM: 0.9422
Average MSE Loss: 210.1854
Average RMSE Loss: 12.5960
```

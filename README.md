
## Cloud Removal

使用卷积神经网络与对抗生成网络，输入有云层遮挡的卫星图片，将云层去除 并 恢复地面。
经过研究与分析，我认为这个任务应该分解为多个子任务：

| 子任务               | 解决方案  | 对应代码     |
| -------------------- | --------- | ------------ |
|提取云层              | 使用U-net | mod_defog.py |
|结合云层，图片去雾    | 使用U-net | mod_mend.py  |
|修补被完全遮盖的地面  | 使用DCGAN | mod_GAN.py   |


然而，在后期实践中我发现：使用经过**小修改的U_net** 可以直接在提取云层的同时，得到去雾后的地面图片，方案简化如下：
| 子任务               | 解决方案  | 对应代码    |
| -------------------- | --------- | ----------- |
| 提取云层，图片去雾   | 使用U-net | mod_mend.py |
| 修补被完全遮盖的地面 | 使用DCGAN | mod_GAN.py  |


### 代码运行

    mod_***.py
    读取数据集文件并开始训练（或者继续训练）
    可以按ctrl + C 传递一个 Keyborad Interrupt 终止训练
    
    mod_defog.py    直接提取云层，并去雾
    mod_mend.py     先提取云层，然后结合云层图像 去雾
    mod_GAN_spot.py 修补图像中间的矩形缺失
    mod_GAN_poly.py 修补图像中不规则的缺失
    
    configure.py    配置文件，记录训练参数
    util            文件夹，存放与数据读取有关的文件
    History         历史文件，用来备份，几乎是个回收站

## -
## -
## 模型解释

![defog_result]()
如上图：
![经过小修改的 U-net]()
如上图：图中自动编码器的转置卷积的输出，一定要有序地经过 batch_norm 和 leakly ReLU

### 为何使用U-net？
比起原版的自动编码器，U-net的每一层转置卷积都得到对称位置上，保留了更多空间信息的张量，使得输出的图片保留了原图的更多位置信息。
将U-net 的输出通道数改为4，分别是 单通道的云层灰度图，三通道的RGB地面去雾图片。这是在loss处做了小修改才得以实现的。

### U-net中，loss处的小修改
我使用了与cycleGAN相同的损失函数，用来衡量两种图片的差异，相减，并取绝对值，那么这就是l1范数（街区距离）。 [tensorflow.losses.absolute_different( )][1]
l2范数我也试过了，没什么差别。
计算云层的loss，我当然是比较 输出的云层，与正确的云层(ground-truth)，然而，计算地面的loss时，我并不是直接比较 输出的地面图片与 正确的地面图片，而是将它们都叠加上正确的云层后，再进行比较。
这就是我提到的**loss处的小修改**，这个修改是基于下面的推断：
当地面被不透明度高于75%的云层遮盖，那么我们难以从残存的像素信息中，恢复出它原来的颜色。
![a simulation of aerial image which covered by clouds of different thicknesses]()
所以我将地面图片叠加上云层后，云层厚度大的区域，即便恢复的情况很差（本来就会很差），也不会反映到loss上去，模型可以更加专注于云层的提取，和地面的去雾工作，云层重度遮盖的区域，就留给GAN去做吧。

而这一部分，我使用DCGAN 完成地面情况的修补
### 为何使用DCGAN？
![DCGAN_model]()
如上图：图中自动编码器的转置卷积的输出，一定要有序地经过 batch_norm 和 leakly ReLU
我在小规模的数据集上，实现并尝试了原版GAN 以及他的变种们： DCGAN、WGAN、WGAN-GP、LS_GAN 以及 Tensor-GAN，这些变种可以在更稳定的训练下，取得比GAN更好的效果，然而，这些变种之间却没有显著的差别。

> #### 关于WGAN
WGAN也是一个很好的模型，拜读原文后，有一个地方是值得我们注意的：WGAN里面用 Wasserstein距离取代了GAN原先的 KL散度、JS散度。
它对输出分布的衡量与以往不同：Wasserstein距离描述的是分布之间的距离，而非real样本与fake样本之间的距离，并且这个Wasserstein距离使用的是一个神经网络定义的函数衡量的，然后加上约束（lipchitz条件)。
参考了： [互怼的艺术：从零直达WGAN-GP - 苏剑林][2]

虽然从WGAN的两篇论文上看，作者用证明了WGAN有种种好处，但是在大规模的训练集上测试后，我选择使用模型简洁，训练时间短的DCGAN.（简而言之，就是效果没有多好，而训练时间却变长了 time_WGAN : time_DCGAN = 3 : 2），也请读者注意，有可能效果不好的原因是我，而非WGAN。

### 对DCGAN 的各种修改
|全称| 下文将使用的代称
|---|---
|对抗生成网络 |GAN
|生成器 |G( ), Generator 
|判别器 |D( ), Discriminator 
|生成器损失 |loss_G
|判别器损失 |loss_D
|真实的图片   | real
|生成的图片   | fake
|缓存的图片   | buff （它也是fake）

数据量小的时候，做图形修补任务，比如在MNIST上，加一条横线，或者删去半边，然后用GAN生成删去的部分，目的是恢复原来的数字。如果数字还能被轻易认出来是数字几，那么就是恢复成功了。
在小的数据集上，GAN以及它的变种们，都可以完成任务，概括地说，越简单的模型，成功恢复图片的训练时间越短。
然而，当训练集增大的时候，当恢复彩色图片的时候，图像修补的对抗训练会出很多问题，下面是我遇到的问题的解决方法。



|问题           | 解决方案 
| ------------- | ---------
|判别器训练过度 |适时暂停对判别器的训练
|生成器多样性低 |同时对局部图片与全局图片进行判别
|判别器不稳定   |额外地 对历史图片也进行判别

#### 1.平衡 生成器与判别器 的训练
训练的时候，过度训练的判别器D(x)，几乎区分出所有real与fake，loss_D 下降地很低，此时生成器G(x) 生成的图片效果差，loss_G 无法为优化提供方向。所以要适当地调节判别器D 与 生成器G 的训练量：每次训练完，统计DCGAN 的两个loss，如果loss_G/loss_D 的比例大于某个值，则停止对判别器的训练，直到比值回到正常范围，反之亦然。（一般是判别器训练过度）
![unblance training logs]()
如图，下面接近0的曲线为loss_D，可以看到，在30步时，训练奔溃，loss无法提供优化方向。与此同时，loss 开始剧烈抖动

#### 2.全局与局部同时判别
这个方法来自于文章[Globally and Locally Consistent Image Completion - SIGGRAPH 2017][3]，我尝试了，好用.
若修补区域不是矩形（比如我的云层），也没关系。我的修改方案是：将修补区外的部分变为0后，也放入判别器进行检测。
![model of Globally and Locally Consistent Image Completion - SIGGRAPH 2017]()

#### 3.额外保存历史图片
训练数据集扩大后，发现判别器会在长时间训练后失效，导致模型震荡(model oscillation) 。
我在cycleGAN 的文章中看到，他用了Shrivastava 的Learning from simulated and unsupervised images through adversarial training 中提及的方法，保存生成器的历史图片，然后与其他图片一同传入判别器进行训练。
可以保证判别器在学习判别新图片的时候，不会忘记历史图片的判别。避免模型震荡。

![gif about the model oscillation, without buffers]()
上图是一张5帧的gif，可以看到第二训练出好效果后，第三帧又变差。后面震荡还在持续。这就是没有使用本页方法的后果。

## -
## -
## 数据集

地面的无云图片，使用的是[法国的无云的卫星图片数据集 (Inria Aerial Image Labeling Dataset)][4]；
云层的图片，使用的是 [美国国家海洋和大气管理局 的纯云层的红外光图像 (Colorized Infrared images)][5] ，我将这些图片根据云层厚度，转化为单通道的灰度图像，并去掉了标示行政区划的白线；
图像修补的时候，图像上的斑块是使用随机顶点的多边形绘制的，因为多边形有时候面积会接近0，我又画了一个固定半径的圆形进去。


## -
## -
## 相关参考
## 卷积算法（卷积 与 转置卷积）
[Convolution arithmetic in Deep Learning](https://github.com/vdumoulin/conv_arithmetic)

[Tensorflow API 讲解——tf.layers.conv2d  HappyRocking的专栏](https://blog.csdn.net/HappyRocking/article/details/80243790)

[TensorFlow中CNN的两种padding方式“SAME”和“VALID”](https://blog.csdn.net/wuzqchom/article/details/74785643)

'valid'(不足，便舍弃): Lnew = ceil((L−F+1)/S)
'same'(不足，便补0):   Lnew = ceil(L/S)

[Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)

## Perceptual Losses for Real-Time Style Transfer and Super-Resolution



## CycleGAN, Unpaired Image-to-Image 

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[CycleGAN HomePage Github.io](https://junyanz.github.io/CycleGAN/)
[CycleGAN arVix, ICCV 2017](https://arxiv.org/pdf/1703.10593.pdf)

[CycleGAN的原理与实验详解 - 何之源](https://zhuanlan.zhihu.com/p/28342644)
[异父异母的三胞胎：CycleGAN, DiscoGAN, DualGAN - 罗若天](https://zhuanlan.zhihu.com/p/26332365)



## Image completion

Globally and Locally Consistent Image Completion - SIGGRAPH 2017
	weburl http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/
	girhub https://github.com/satoshiiizuka/siggraph2017_inpainting
Context Encoders: Feature Learning by Inpainting
	arXiv paper https://arxiv.org/pdf/1604.07379.pdf
	GitHb code https://github.com/pathak22/context-encoder
lafarren.com Image Completion
	lafarren.com: paper http://lafarren.com/image-completer/
	GitHub C++ https://github.com/darrenlafreniere/lafarren-image-completer
Keras Image OutPainting	
	GitHub https://github.com/bendangnuksung/Image-OutPainting
	量子位 https://zhuanlan.zhihu.com/p/41114883
GitHub Lightweight_cGANs https://github.com/adamstseng/general-deep-image-completion
GitHub topics Image completion https://github.com/topics/image-completion?l=python
DCGAN tensorflow https://github.com/saikatbsk/ImageCompletion-DCGAN



  [1]: https://link.zhihu.com/?target=https://www.tensorflow.org/api_docs/python/tf/losses/absolute_difference
  [2]: https://spaces.ac.cn/archives/4439
  [3]: http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/
  [4]: https://link.zhihu.com/?target=https://project.inria.fr/aerialimagelabeling/
  [5]: https://link.zhihu.com/?target=https://www.nesdis.noaa.gov/content/imagery-and-data
  [6]: https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown
  [7]: https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#cmd-markdown-高阶语法手册
  [8]: http://weibo.com/ghosert
  [9]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
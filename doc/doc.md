# 作业1：蛋白质二级结构预测

代码实现：[本仓库](https://github.com/Chengyuan-artist/seq_ssp_prediction)

数据样例：

```
{'seq': 'AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR',
 'ssp': 'CCCCCCHHHHHHHHHHHCCCCCCEECHHHHHHHCCCHHHCCHHHHCCCCEEEEEEECCCCCCCCCCCCCCEEEEECCCCCCCCCCCEEEEECCCCEEEECCCCCCCEECC'}
```

输入字母表：20个氨基酸；输出字母表：3种二级结构。

3000个蛋白质序列，最短长度为60，最长长度为250，平均长度为148.6。

输入输出如何编码？如何实现变长输入输出？

## 网络输入输出

本作业采用one-hot编码输入输出。

思路1：固定输入输出为最大长度250。

则：

| 模型     | 输入张量形状  | 输出张量形状 |
| -------- | ------------- | ------------ |
| MLP      | (N, 250 * 20) | (N, 250 * 3) |
| ResNet1d | (N, 20, 250)  | (N, 250 * 3) |

N为训练样本数量。

这种方式经过实验后发现模型(mlp和resnet1d)训练后准确率均在39%左右，而且loss曲线没有明确拐点，推测模型处于欠拟合状态。



思路2：按固定长度切分训练样本，输入输出长度固定为切分长度

split_len 为切分长度

| 模型     | 输入张量形状        | 输出张量形状       |
| -------- | ------------------- | ------------------ |
| MLP      | (N, split_len * 20) | (N, split_len * 3) |
| ResNet1d | (N, 20, split_len)  | (N, split_len * 3) |

本实验采取思路2并取split_len = 20。为了统一的对比标准，MLP和ResNet1d均取相同的split_len。

## 数据处理

为了方便分割序列的数据处理，`src/dataset.py`中实现了一系列方便的处理函数：

```
def split_seq(seq, split_len = 250):
```

将字符串序列分割成`split_len`的子字符串

```
def seq_to_onehot(seq, is_ssp = False, reqular_len = 250):
```

将字符串转换为形状为(regular_len, 3) 或 (regular_len, 20)的二维one-hot张量

```
def onehot_to_sequence(one_hot, is_ssp=False):
```

将one-hot张量转换为序列

```
def prob_to_onehot(prob):
```

将网络输出结果（概率张量）转化为one_hot张量

## 样本划分

本实验仅进行了训练集和验证集划分，在训练开始前将样本进行随机划分为80%训练集，20%验证集。

测试集和验证集数据类型不同。测试集后续需要进行切分，输入输出均为张量。验证集则分别为seq和ssp字符串，在推理时对单个数据进行切分，转化为one-hot张量并叠加后送入网络。

## 网络推理

由于网络输入输出被固定为split_len，因此网络只能预测长度为split_len序列的预测结果。

由于实验需要统计Q3准确率，即每个序列的预测准确率，因此推理需要一次预测一条序列并给出正确率。

```python
         with torch.no_grad():
                q3s = []
                for data in self.valid_loader:
                    inputs, targets = data[0], data[1]  // A batch of data
                    for seq, ssp in zip(inputs, targets): // single data (seq, ssp) 

                        split_seq = ProteinDataset.split_seq(seq, self.split_len) // split seq
                        seq_tensors = [ProteinDataset.seq_to_onehot(seq, reqular_len=self.split_len).view(-1) for seq in split_seq] // transfrom every split seq to one-hot tensor 
                        input_tensor = torch.stack(seq_tensors) // stack these one-hot tensors
                        input_tensor = input_tensor.to(self.device)
                        # print(input_tensor.shape)
                        output_tensor = self(input_tensor) // put the stacked tensor into the network
                        # print(output_tensor.shape)

                        one_hot = ProteinDataset.prob_to_onehot(output_tensor.view(-1, 3))
                        # print(one_hot.shape)
                        ssp_predicted = ProteinDataset.onehot_to_sequence(one_hot, is_ssp=True)
                        
                        // compare ssp with ssp_predicted to cal Q3 acuarrncy
```

处理方式见上述注释。

## 结果

### mlp

网络结构为

```
Linear(20 * 20, 32) -> ReLu -> Linear(32, 20 * 3) -> softmax(20, 3)

input:torch.Size([19050, 400])
output:torch.Size([19050, 60])
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1            [-1, 19050, 32]          12,832
              ReLU-2            [-1, 19050, 32]               0
            Linear-3            [-1, 19050, 60]           1,980
           Softmax-4                [-1, 20, 3]               0
================================================================
```

两层全连接(hidden取32)，一层ReLu，最后一层softmax对每3个输出做softmax，即每个位置的三个类别预测做softmax。

<img src="images/mlp_loss.png" style="zoom: 80%;" />

Valid data len: 600

Average q3 accuracy: 59.52078323115891%

Mean squared error: 0.7551768759005673%

最终在验证集上的准确率为 59.52 ± 0.76 %

若将模型tain的verbose设为True, 则可看到每个序列的预测结果以及Q3准确率

### resnet1d

实现代码更改自[resnet1d](https://github.com/hsd1503/resnet1d)`test_physionet.py`, 将resnet的输入输出参数更改，use_bn设为False, 其他参数未更改。模型为ResNet50。

![](images/resnet1d_loss.png)

Valid data len: 600                                                                              

Average q3 accuracy: 63.74675038164278%

Mean squared error: 0.7119044158372616%

最终在验证集上的准确率为 63.75 ± 0.71%

相比mlp准确率提升4.23%
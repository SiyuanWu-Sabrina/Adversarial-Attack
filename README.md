# Experiment

## 文件结构
- attack：存放各类攻击算法源代码及相关文档
- datasets：存放数据集
- checkpoints：存放训练好的模型参数
  - 存储格式：modelName_dataset_epochNum_extraInfo.pt or .tar
- models: 存放模型代码
- output：模型训练过程中的命令行输出
- result：模型攻击ASR，范数结果
- 用于训练models的代码（models本身在./models中），以及用于测试攻击效果的代码
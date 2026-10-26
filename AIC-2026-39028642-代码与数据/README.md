# AIC-2026-39028642 复现说明

## 环境与文件完整性

本目录提供本作品的训练、输入生成、推理、检查程序及必要权重。上一层最终模型文件和结果ZIP须保持与本目录同层关系。数据原图由复现人员自行放置；不要把自己的原图、生成像素缓存、特征缓存或训练输出混入提交材料根目录。

推荐使用environment.json记录的Python、PyTorch、torchvision、transformers、NumPy和Pillow版本；requirements.txt给出安装依赖。CUDA驱动须兼容所装PyTorch。提供最终权重的预测不需要下载官方初始化权重；重新训练才调用fetch_initialization.py。

```bash
python -m pip install -r requirements.txt
python validate_reproduction.py
```

该检查严格核对正式模型SHA256、结果ZIP内CSV、37444个唯一文件名、固定顺序和0000—0749四位类别编号。加载模型的校验失败时应重新下载原材料，不能修改校验值掩盖损坏。

## 数据放置与完整检查

官方TRAIN共148695张、750类，目录名0000—0749；官方TEST共37444张。TRAIN相对路径及给定标签由data/train_index.json规定，TEST名称和顺序由data/test_names.json规定。图像必须保留原文件名、内容和所在类别；不要重新编码图像或自行改变标签映射。

```text
data/official/
├─ train/
│  ├─ 0000/ 原始图像文件
│  ├─ ...
│  └─ 0749/ 原始图像文件
└─ test/ 原始图像文件
```

```bash
python validate_reproduction.py --split train
python validate_reproduction.py --split test
```

按清单缺失任何一张图像都会报错。没有额外标注验证集，全部TRAIN参加训练。TRAIN分类诊断不等于独立验证准确率。TEST图像在正式推理中形成完整查询图；不能拆成小子集各自图推理后再拼接结果。

## 使用交付模型复现正式预测

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh ./pred_results.csv
python validate_reproduction.py --prediction ./pred_results.csv
```

run.sh先校验原图，build_cache.py生成512像素中心及完整画幅缓存，模型生成特征和配置规定的分类视角，再运行固定统计读出及查询图步骤，最后输出两列CSV。center与complete画幅、插值位置、归一化顺序、FP16缓存、BF16模型前向和批量大小均属于数值复现设置，不能擅自替换。完整CSV校验检查的是随包模型的正式结果。

本目录的小型固定推理统计文件承载训练类统计、查询邻居或清单，是随最终模型冻结的推理输入；不含TEST真值。生成的大型图像与模型特征缓存不需要上传。缓存由代码位置解析，不依赖作者电脑目录。

## 从官方初始化重建训练

技术路线：官方 CLIP ViT-B/32 初始化 → 384 像素上下文适配与投影补空间训练，按8轮预算调度、完成7轮 → Fisher 轴约束训练400次更新 → TRAIN 双空间邻域软目标生成 → 缓存分类修正512次更新 → 补空间原型读出512次更新。

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh
python validate_training.py --model runs/full_training/retrained_model.pt
```

train.sh调用reproduce_training.py，完整连接原图检查、官方初始化和特征生成、全部记录的视觉训练阶段、TRAIN统计和读出拟合。配置文件：training/visual/initial_config.json、training/fisher_parameters.json、training/cached_parameters.json、training/null_parameters.json。每个阶段保留自己的完整学习率调度预算；阶段的停止位置与预算不是一回事，不得将短程诊断当作正式训练。

初始化固定下载openai/clip-vit-base-patch32的指定修订到resources/weights/clip-vit-base-patch32。初始视觉特征来自官方初始化对TRAIN的224像素FP32前向、归一化和FP16保存；训练增强按本作品源码及配置执行。模型参数训练采用BF16。首阶段输出和后续检查点均在本目录runs/，完整最终产物在runs/full_training/retrained_model.pt。总入口不会覆盖上一层交付最终模型。

后续机制使用此次训练生成的前驱。统计拟合只读TRAIN图像和官方给定标签：原始文件字节内容组、模型置信度、五折邻域一致性、同内容重复密度，以及本算法使用的类别几何量。reliability与density不是新的人工标注；邻域拟合排除同一内容组所在折，TEST标签不参与。读出输入和软目标通过对应脚本重建，不能直接把正式最终模型的已拟合统计当作此次新训练的输入。

validate_training.py严格加载完整模型结构，检查浮点张量有限性；可选--compare逐张量比较对应模型。跨软件、硬件和随机流的重新训练可能产生数值差异，不能从脚本成功推断参数逐位相同。正式结果的确定性核验应使用提供的最终权重与固定推理参数。

## 短程诊断和失败恢复

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh --max-steps 2
```

该选项仅检查视觉阶段能运行和检查点衔接，保留完整调度预算并提前停止，不完成统计读出、不复现正式成绩。同名runs输出已存在时入口停止；重新训练前将自己的旧输出移到本目录之外。输入缓存若存在且完整会复用，图像更换、路径顺序或模型变化后须删除对应生成缓存再重建，不能继续使用旧特征。

CUDA内存不足时先释放自己启动的程序；随包最终预测的分类批量不能为省显存擅自调整，否则可能改变数值与边界预测。训练调参需要另外记录参数。下载初始化失败可修复网络后重新运行；缺图、解码或校验报错应先修复数据，而不能跳过样本或改写校验值。

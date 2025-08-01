本项目是作品“基于safeear的深度伪造检测应用”的后端文件，提供基于 Node.js 的在线推理服务。

---

## 环境依赖

- python 3.9
- python依赖：requirements.txt
- ffmpeg
- Node.js (v14+) 与 npm

---

## 安装与使用

1. 进入目录：
   ```bash
   cd SafeEar
   ```

2. 安装 Python 依赖：
   ```bash
   pip install torch pytorch-lightning hydra-core fairseq librosa soundfile tqdm
   pip install -r requirements.txt
   ```

3. 安装 Node.js 依赖：
   ```bash
   npm install express multer
   ```

4. 安装ffmpeg

   ```
   apt install ffmpeg
   ```

---

## 数据准备

1. 下载 ASVspoof 数据集，并将 `train.tsv`、`dev.tsv`、`eval.tsv` 放入 `datas/ASVSpoof2019/`。
2. 将对应的 FLAC 音频放入 `datas/audio/flac/`，原始音频放入 `datas/audio/orig/`。
3. 提取 HuBERT 特征：
   ```bash
   python datas/dump_feature.py \
     --ckpt /path/to/hubert_checkpoint.pt \
     --input_dir datas/audio/orig/ \
     --output_dir datas/audio/processed/ \
     --layer <layer_index>
   ```

---

## 模型训练

- ASVspoof2019：
  ```bash
  python train.py --config-name train19
  ```

训练日志与检查点保存在 `Exps/ASVspoof19/lightning_logs/`。

---

## 模型评估

```bash
python test.py --config-name train19
```
评估结果将记录在 `score.csv`，处理状态记录在 `cached_file.csv`。

### 本地单文件推理

1. **提取音频特征**

   ```
   python datas/dump_feature.py --audio_path datas/audio/file_name.mp3
   ```

   > 默认会将生成的特征 `.npy` 文件保存在 `datas/audio/processed/`。

2. **执行推理脚本**

   ```
   python test.py 
   ```

3. **查看推理结果**

   推理分数会写入 `score.csv`，每行格式为 `file_path,probability,pred_label`。

---

## 在线推理服务器

1. 在 `server.js` 中配置 TLS 证书路径。
2. 启动服务器：
   ```bash
   node server.js
   ```

服务器监听 3006 端口，提供以下接口：

- `POST https://whusafeear.top/common/upload`：上传音频文件进行推理,服务器返回推理结果。

### 通过服务器进行单个音频推理示例

通过 cURL 上传单个音频文件并获取推理结果，例如：

```
curl -k -F "file=@/path/to/audio.flac" https://whusafeear.top/common/upload
```

该请求将返回NDJSON，每行示例如下：

```json
{
    "message": "detect success",
    "modelLogits": 0,
    "probability": 0.003186,
    "predictedLabel": 1
}
```

---

## 配置说明

所有超参数、数据路径等可在 `config/` 下的 YAML 文件中修改，通过 Hydra 自动管理。

---

## 仓库结构

```
SafeEar/
├── config/                       # Hydra 配置文件
│   ├── train19.yaml              # ASVspoof2019 训练配置
│   └── train21.yaml              # ASVspoof2021 训练配置
├── datas/                        # 数据处理与特征提取
│   ├── ASVSpoof2019/             # 数据集元信息（train.tsv、dev.tsv、eval.tsv）
│   ├── audio/
│   │   ├── orig/                 # 原始音频文件
│   │   ├── flac/                 # FLAC 格式音频
│   │   └── processed/            # 提取的特征（.npy）
│   ├── dump_feature.py           # HuBERT 特征提取脚本
│   └── feature_utils.py          # 特征处理工具
├── safeear/                      # 核心代码包
│   ├── datas/                    # LightningDataModule 定义
│   ├── models/                   # 模型结构（decouple、discriminator、quantization）
│   ├── losses/                   # 自定义损失函数
│   ├── trainer/                  # 训练逻辑（SafeEarTrainer）
│   └── utils/                    # 辅助工具
├── Exps/                         # 实验输出（检查点、日志、指标）
│   └── ASVspoof19/
├── train.py                      # 训练入口脚本（Hydra）
├── test.py                       # 评估脚本，带缓存功能
├── server.js                     # Express.js 推理服务器
├── cached_file.csv               # 缓存文件，用于跟踪处理记录
├── score.csv                     # 评估分数记录
└── tree.txt                      # 项目文件树（生成）
```

## 引用说明

本项目基于 [LetterLiGo/SafeEar](https://github.com/LetterLiGo/SafeEar) 进行开发。

```
@misc{LetterLiGo_SafeEar,
  author       = {LetterLiGo},
  title        = {SafeEar: Audio Anti-Spoofing Detection Framework},
  howpublished = {\url{https://github.com/LetterLiGo/SafeEar}},
  year         = {2025}
}
```
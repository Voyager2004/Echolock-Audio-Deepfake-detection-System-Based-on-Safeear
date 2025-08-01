# EchoLock 项目

## 项目简介
EchoLock是基于SafeEar项目开发的工程化落地应用，专注于音频深度伪造检测的实际应用场景。本项目将SafeEar的核心技术进行了工程化改造，提供了完整的前后端解决方案，适用于实际生产环境。

## 基于项目
本项目基于以下开源项目开发：
- **SafeEar**: [https://github.com/AI-secure/SafeEar](https://github.com/AI-secure/SafeEar)
- 感谢SafeEar团队提供的优秀音频深度伪造检测算法基础

## 项目特色
- 🔧 **工程化设计**: 基于SafeEar核心算法，进行了全面的工程化改造
- 📱 **多端支持**: 支持鸿蒙APP和微信小程序双端应用
- 🚀 **生产就绪**: 优化了性能和稳定性，适合实际部署
- 🎯 **场景化应用**: 针对实际使用场景进行了功能定制

## 项目结构
```
├── 前端源代码/
│   ├── HarmonyOS-APP/          # 鸿蒙应用端
│   └── 微信小程序/              # 微信小程序端（uni-app框架）
└── 后端源代码/
    └── Backend/                # Node.js后端服务
```

## 技术栈
- **前端**: 
  - 鸿蒙APP (ArkTS)
  - 微信小程序 (uni-app + Vue.js)
- **后端**: Node.js + Express
- **AI模型**: 基于SafeEar优化的Python + PyTorch模型
- **部署**: 支持Docker容器化部署

## 快速开始

### 环境要求
- Node.js >= 14.0
- Python >= 3.8
- npm 或 yarn

### 模型权重获取
⚠️ **重要说明**: 本仓库不包含预训练模型权重文件

由于模型文件过大（约726MB），预训练权重文件未包含在此仓库中。如需获取完整的模型权重文件，请：

1. **从SafeEar原项目获取**: 访问 [SafeEar官方仓库](https://github.com/AI-secure/SafeEar) 下载原始模型
2. **联系项目维护者**: 通过Issues或邮件联系获取适配版本的模型权重
3. **自行训练**: 使用提供的训练脚本和数据集重新训练模型

模型文件应放置在以下位置：
```
后端源代码/Backend/safeear/best.ckpt
```

### 后端启动
```bash
cd 后端源代码/Backend
npm install

# 安装Python依赖
pip install -r requirements.txt

# 确保模型权重文件已放置在正确位置
# 启动服务
npm start
```

### 前端开发

#### 微信小程序
```bash
cd 前端源代码/微信小程序
# 使用HBuilderX或其他uni-app开发工具打开项目
# 配置小程序开发者工具进行预览和调试
```

#### 鸿蒙APP
```bash
cd 前端源代码/HarmonyOS-APP
# 使用DevEco Studio打开项目
# 配置鸿蒙开发环境进行编译和调试
```

## 核心功能
- 🎵 **音频深度伪造检测**: 基于SafeEar算法的高精度音频内容检测
- 📊 **实时分析**: 支持实时音频流分析和批量文件处理
- 🔍 **多格式支持**: 支持多种音频格式的检测和分析
- 📈 **结果可视化**: 提供直观的检测结果展示和统计分析
- 🔐 **安全保障**: 完善的数据安全和隐私保护机制

## 部署说明

### 开发环境
```bash
# 克隆项目
git clone https://github.com/你的用户名/EchoLock.git
cd EchoLock

# 安装后端依赖
cd 后端源代码/Backend
npm install

# 安装Python依赖
pip install -r requirements.txt

# ⚠️ 重要：配置模型权重文件
# 请确保将模型权重文件放置在 safeear/best.ckpt

# 启动服务
npm start
```

## 与SafeEar的关系
EchoLock项目基于开源的SafeEar音频检测算法，在以下方面进行了工程化改进：
- **性能优化**: 针对实际应用场景优化了算法性能
- **接口标准化**: 提供了标准化的API接口
- **多端适配**: 开发了移动端和Web端应用
- **部署简化**: 简化了部署流程，支持一键部署
- **监控告警**: 增加了系统监控和告警功能

### 引用声明
如果您在学术研究中使用了本项目，请同时引用SafeEar原始论文：
```
@inproceedings{safeear2023,
  title={SafeEar: Content Privacy-Preserving Audio Deepfake Detection},
  author={Zhou, Shuhang and Wang, Xiaoyi and Liu, Zihan and Deng, Xiaowei and Zhu, Qi and Yang, Zhiyao and Xu, Yifan and Liu, Siwei},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## 文件说明
- `后端源代码/Backend/safeear/`: SafeEar核心算法实现
- `后端源代码/Backend/server.js`: Web服务器入口
- `前端源代码/HarmonyOS-APP/`: 鸿蒙应用源码
- `前端源代码/微信小程序/`: 微信小程序源码
- `requirements.txt`: Python依赖列表
- `package.json`: Node.js依赖配置

## 注意事项
- ⚠️ **模型权重**: 本仓库不包含预训练模型权重文件，需要单独获取
- 🔒 **数据安全**: 生产环境部署前请仔细阅读安全配置指南
- 📝 **许可协议**: 使用时请遵守SafeEar原项目的许可协议
- 🐛 **问题反馈**: 遇到问题请先查看SafeEar原项目文档

## 开发团队
本项目由[团队名称]基于SafeEar开源项目进行工程化开发。

## 许可证
本项目遵循 [MIT License](./LICENSE)，同时尊重SafeEar项目的相关许可协议。

## 致谢
- 感谢SafeEar团队提供的优秀开源项目基础
- 感谢所有为音频安全检测技术发展做出贡献的研究者
- SafeEar原项目: https://github.com/AI-secure/SafeEar
---

> EchoLock团队出品

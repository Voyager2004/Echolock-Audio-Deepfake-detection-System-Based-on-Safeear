<template>
  <view class="container">

    <!-- 时间显示 -->
    <view class="time-display">
      <text class="time">{{ timeDisplay?.h||'00' }}</text>
      <text class="colon">:</text>
      <text class="time">{{ timeDisplay?.m||'00' }}</text>
      <text class="colon">:</text>
      <text class="time">{{ timeDisplay?.s||'00' }}</text>
    </view>

    <!-- 波形可视化 -->
    <!--    <view class="visualizer" v-if="state !== 0">
      <view v-for="(bar, index) in visualizerBars" :key="index" class="bar" :style="{ height: bar.height + 'px' }">
      </view>
    </view> -->

    <!-- 控制按钮组 -->
    <view class="controls">
      <!-- 播放按钮 -->
      <button class="control-btn play-btn" :class="{ disabled: !tempFilePath }" @tap="playRecording">
        <uni-icons type="headphones" size="30" color="blue"></uni-icons>
      </button>

      <!-- 主录音按钮 -->
      <view class="main-btn" :class="{ recording: state === 1, paused: state === 2 }" @tap="handleRecord">
        <view class="inner-circle"></view>
      </view>

      <!-- 停止按钮 -->
      <button class="control-btn stop-btn" :class="{ disabled: state === 0 }" @tap="stopRecording">
        <uni-icons type="circle-filled" size="30" color="red"></uni-icons>
      </button>
    </view>

    <!-- 状态提示 -->
    <view class="status-text">
      {{ statusMessage }}
    </view>
    <view style="display:flex">
      <button class="action-btn" :disabled="!tempFilePath" @tap="uploadFile">
        <uni-icons type="cloud-upload" size="24" color="#666"></uni-icons>
        <text>云端检测</text>
      </button>

    </view>
    <view style="margin-top: 30rpx;">
      <text style="font-size: 28rpx; color: #2c3e50;">云端检测伪造概率：<text
          :style="{color: probabilityColor,'font-size':'36rpx'} ">{{result}}%</text></text>
    </view>
    <!-- 添加uniapp进度条 -->
    <view style="width: 80%; margin-top: 20rpx;">
      <progress :percent="result" stroke-width="12px" :activeColor="probabilityColor" backgroundColor="#e5e5e5"
        border-radius="6px" active />
    </view>

    <!-- 添加步骤条 -->
    <view class="steps">
      <uni-steps :options="stepOptions" :active="currentStateIndex" active-color="#007AFF"
        direction="column"></uni-steps>
    </view>
  </view>
</template>

<script setup>
  import {
    ref,
    reactive,
    onUnmounted,
    computed
  } from 'vue'
  import {
    onShow,
    onLoad
  } from '@dcloudio/uni-app';
  // 录音管理器
  const tape = uni.getRecorderManager()
  // 音频上下文
  const audio = uni.createInnerAudioContext()

  // 响应式状态
  const state = ref(0) // 0:停止 1:录音中 2:暂停
  const time = ref(0)
  const tempFilePath = ref(null)
  const timer = ref(null)
  const result = ref(0)

  // 时间显示
  const timeDisplay = reactive({
    h: '00',
    m: '00',
    s: '00'
  })
  onLoad(() => {
    updateTime()
  })
  // 自动清理定时器
  onUnmounted(() => {
    clearInterval(timer.value)
    audio.destroy()
  })

  // 格式化时间
  const formatTime = (time) => {
    return time < 10 ? `0${time}` : time
  }

  // 更新时间显示
  const updateTime = () => {
    if (!timeDisplay) {
      console.error('timeDisplay is undefined')
      return
    }

    const total = time.value || 0
    timeDisplay.h = formatTime(Math.floor(total / 3600))
    timeDisplay.m = formatTime(Math.floor((total % 3600) / 60))
    timeDisplay.s = formatTime(Math.floor(total % 60))
  }

  // 开始/暂停录音
  const handleRecord = () => {
    console.log('handleRecord')
    switch (state.value) {
      case 0: // 开始录音
        startRecording()
        break
      case 1: // 暂停录音
        pauseRecording()
        break
      case 2: // 继续录音
        resumeRecording()
        break
    }
  }

  // 开始录音
  const startRecording = () => {
    state.value = 1
    tape.start()
    startTimer()

  }

  // 暂停录音
  const pauseRecording = () => {
    state.value = 2
    clearInterval(timer.value)
    tape.pause()
  }

  // 继续录音
  const resumeRecording = () => {
    state.value = 1
    tape.resume()
    startTimer()
  }

  // 停止录音
  const stopRecording = () => {
    state.value = 0
    clearInterval(timer.value)
    tape.stop()
    resetTimer()
    uni.showToast({
      title: '保存成功',
      duration: 1000
    })
  }

  // 播放录音
  const playRecording = () => {
    if (!tempFilePath.value) {
      uni.showModal({
        title: '提示',
        content: '请先录制音频'
      })
      return
    }
    audio.src = tempFilePath.value
    audio.play()
    startarti.showToast({
      title: '播放中',
      duration: 1000
    })
  }

  // 定时器逻辑
  const startTimer = () => {
    timer.value = setInterval(() => {
      time.value++
      updateTime()
    }, 1000)
  }

  // 重置计时器
  const resetTimer = () => {
    time.value = 0
    updateTime()
  }

  // 录音结束回调
  tape.onStop((res) => {
    tempFilePath.value = res.tempFilePath
  })

  // 计算按钮文本
  const buttonText = computed(() => {
    return ['开始录音', '暂停录音', '继续录音'][state.value]
  })

  // 新增可视化数据
  // const visualizerBars = ref(
  //   Array.from({
  //     length: 15
  //   }, () => ({
  //     height: Math.random() * 30 + 10
  //   }))
  // )

  // 状态提示消息
  const statusMessage = computed(() => {
    return [
      '点击红色按钮开始录音',
      '录音中...',
      '已暂停'
    ][state.value]
  })

  // 播放状态
  const isPlaying = ref(false)

  const uploadFile = async () => {
    if (!tempFilePath.value) return
    startProgress()
    uni.showLoading({
      title: '上传中...',
      mask: true
    })
  
    try {
      console.log(tempFilePath.value)
      const res = await uni.uploadFile({
        url: process.env.VUE_APP_API_BASE_URL + '/common/upload', // 使用环境变量替代硬编码URL
        filePath: tempFilePath.value,
        name: 'file',
        formData: {
          'content-type': 'multipart/form-data'
        },
      })
      console.log(res)
      const receive = JSON.parse(res.data)
      console.log(receive)
      result.value = Number((parseFloat(receive.probability) * 100).toFixed(2))

      // 保存到历史记录
      saveHistory({
        fileName: '录音文件',
        source: '录音上传',
        result: result.value,
        time: new Date().toLocaleString()
      })
      stopProgress()
    } catch (error) {
      uni.showModal({
        title: '上传失败',
        content: error || 'fail',
        showCancel: false
      })
    } finally {
      uni.hideLoading()
    }
  }

  // 保存历史记录
  const saveHistory = (data) => {
    let history = uni.getStorageSync('detectHistory') || []
    history.unshift(data)
    if (history.length > 50) history = history.slice(0, 50)
    uni.setStorageSync('detectHistory', history)
  }

  // 仿照index的颜色计算逻辑
  const probabilityColor = computed(() => {
    const green = [0, 255, 0];
    const red = [255, 0, 0];
    const r = Math.round(red[0] * (result.value / 100) + green[0] * (1 - result.value / 100));
    const g = Math.round(red[1] * (result.value / 100) + green[1] * (1 - result.value / 100));
    const b = Math.round(red[2] * (result.value / 100) + green[2] * (1 - result.value / 100));
    const hex = (r << 16 | g << 8 | b).toString(16).padStart(6, "0");
    return `#${hex}`;
  });

  // 从 index.vue 复制过来的步骤条相关数据和方法
  let timerForSteps = null
  const stepOptions = [{
      title: '请选择音频文件'
    },
    {
      title: '音频加密上传中'
    },
    {
      title: '音频信号预处理中'
    },
    {
      title: '加载深度伪造检测模型中'
    },
    {
      title: '声学特征检测中'
    },
    {
      title: '音频检测完成'
    }
  ]

  let currentStateIndex = ref(0)

  const startProgress = () => {
    currentStateIndex.value = 0
    timerForSteps = setInterval(() => {
      if (currentStateIndex.value < stepOptions.length - 2) {
        currentStateIndex.value++
      }
    }, 3000)
  }

  const stopProgress = () => {
    clearInterval(timerForSteps)
    currentStateIndex.value = stepOptions.length - 1
  }

  onUnmounted(() => {
    clearInterval(timerForSteps)
  })
</script>

<style lang="scss" scoped>
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100vh;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 40rpx;
  }

  .title {
    font-size: 48rpx;
    color: #2c3e50;
    letter-spacing: 4rpx;
    margin: 60rpx 0;
    font-weight: 500;
  }

  .time-display {
    display: flex;
    align-items: baseline;
    margin: 40rpx 0;

    .time {
      font-size: 80rpx;
      color: #2c3e50;
      text-shadow: 2rpx 2rpx 4rpx rgba(0, 0, 0, 0.1);
    }

    .colon {
      font-size: 60rpx;
      margin: 0 10rpx;
      color: #666;
    }
  }

  .visualizer {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    height: 150rpx;
    margin: 60rpx 0;
    gap: 8rpx;

    .bar {
      width: 16rpx;
      background: #409EFF;
      border-radius: 8rpx;
      transition: height 0.2s ease;
    }
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 80rpx;
    margin-top: 60rpx;
  }

  .main-btn {
    width: 160rpx;
    height: 160rpx;
    border-radius: 50%;
    background: #ff4444;
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 8rpx 20rpx rgba(255, 68, 68, 0.3);
    transition: all 0.3s ease;
    position: relative;

    &.recording {
      animation: pulse 1.5s infinite;

      .inner-circle {
        width: 60rpx;
        height: 60rpx;
        border-radius: 12rpx;
      }
    }

    &.paused {
      background: #ffa500;

      .inner-circle {
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 30rpx 0 30rpx 50rpx;
        border-color: transparent transparent transparent #fff;
      }
    }
  }

  .inner-circle {
    width: 80rpx;
    height: 80rpx;
    background: #fff;
    border-radius: 50%;
    transition: all 0.3s ease;
  }

  .control-btn {
    width: 100rpx;
    height: 100rpx;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 4rpx 12rpx rgba(0, 0, 0, 0.1);
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.2s ease;

    &.disabled {
      opacity: 0.5;
      filter: grayscale(1);
    }

    .icon {
      width: 50rpx;
      height: 50rpx;
    }
  }

  .status-text {
    margin: 20rpx;
    color: #666;
    font-size: 28rpx;
  }

  @keyframes pulse {
    0% {
      transform: scale(1);
    }

    50% {
      transform: scale(1.05);
    }

    100% {
      transform: scale(1);
    }
  }

  .steps {
    margin-top: 30rpx;
    width: 60%;
  }
  
  .action-btn{
    border-radius: 50rpx;
  }
</style>
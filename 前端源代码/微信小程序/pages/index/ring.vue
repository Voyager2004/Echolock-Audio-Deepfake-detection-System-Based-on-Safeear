<template>
  <view class="progress-circle" :style="{ width: size + 'rpx', height: size + 'rpx' }">
    <view class="outring" :style="outringStyle">
      <!-- 进度条 -->
      <view class="progress-bar" v-if="showProgress" :style="progressBarStyle"></view>
      <!-- 小球 -->
      <view class="ball start" v-if="showProgress" :style="startBallStyle"></view>
      <view class="ball end" v-if="showProgress" :style="endBallStyle"></view>
      <view class="ball defaultball" v-if="showDefaultBall" :style="defaultBallStyle"></view>
    </view>
    <!-- 内环 -->
    <view class="innerring" :style="innerringStyle"></view>
    <view class="custom-label" :style="{color:progressColor[progressColor.length - 1]}">
      {{ percent }}%
    </view>
  </view>
</template>

<script>
  export default {
    name: 'ProgressCircleBar',
    props: {
      // 整体尺寸（单位rpx）
      size: {
        type: Number,
        default: 460
      },
      // 进度百分比（0-100）
      percent: {
        type: Number,
        default: 0,
        validator: value => value >= 0 && value <= 100
      },
      // 进度条颜色（支持渐变色）
      progressColor: {
        type: [String, Array],
        default: () => ['#bced4f', '#99e252', '#72d555']
      },
      // 内环背景色
      innerColor: {
        type: String,
        default: 'transparent'
      },
      // 小球尺寸（单位rpx）
      ballSize: {
        type: Number,
        default: 100
      }
    },
    computed: {
      // 显示进度条的条件
      showProgress() {
        return this.percent >= 5
      },
      // 显示默认小球的条件
      showDefaultBall() {
        return this.percent > 0 && this.percent < 5
      },
      // 外环样式
      outringStyle() {
        return {
          background: `rgba(0,0,0,0.1)`
        }
      },
      // 进度条样式
      progressBarStyle() {
        const endAngle = (this.percent / 100) * 360
        const colors = Array.isArray(this.progressColor) ?
          this.progressColor :
          [this.progressColor]
        let gradient = 'conic-gradient('
        colors.forEach((color, index) => {
          const stop = (endAngle * (index / (colors.length - 1))).toFixed(2)
          gradient += `${color} ${stop}deg${index < colors.length - 1 ? ',' : ''}`
        })
        gradient += `, transparent ${endAngle}deg)`

        return {
          background: gradient,
          width: this.size + 'rpx',
          height: this.size + 'rpx'
        }
      },
      // 内环样式
      innerringStyle() {
        const innerSize = this.size * 0.7 // 按比例计算内环尺寸
        return {
          width: innerSize + 'rpx',
          height: innerSize + 'rpx',
          backgroundColor: this.innerColor
        }
      },
      // 起始小球样式
      startBallStyle() {
        return {
          width: this.ballSize + 'rpx',
          height: this.ballSize + 'rpx',
          top:this.ballSize/2+'rpx',
          'z-index':'0',
          backgroundColor: this.progressColor[0]
        }
      },
      // 结束小球样式
      endBallStyle() {
        const angle = ((this.percent / 100) * 360 - 90) * Math.PI / 180
        const radius = (this.size - this.ballSize) / 2
        const center = this.size / 2
        const c=this.progressColor[this.progressColor.length - 1]
        console.log(c)

        const x = center + radius * Math.cos(angle)
        const y = center + radius * Math.sin(angle)

        return {
          width: this.ballSize + 'rpx',
          height: this.ballSize + 'rpx',
          left: x + 'rpx',
          top: y + 'rpx',
          'z-index':'1',
          backgroundColor: c,
        }
      },
      // 默认小球样式
      defaultBallStyle() {
        return {
          width: this.ballSize + 'rpx',
          height: this.ballSize + 'rpx',
          background: `linear-gradient(90deg, 
          ${this.progressColor[0]} 0%, 
          ${this.progressColor[this.progressColor.length - 1]} 100%)`
        }
      }
    }
  }
</script>

<style lang="scss" scoped>
  .progress-circle {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    .custom-label {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 50rpx;
      color: #222;
      z-index: 20;
      font-weight: bold;
    }

    .outring {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      position: relative;
      overflow: hidden;
    }

    .innerring {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      z-index: 4;
      box-shadow: 0 0 20rpx rgba(0, 0, 0, 0.1);
    }

    .progress-bar {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      z-index: 3;
    }

    .ball {
      position: absolute;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      z-index: 5;
      box-shadow: 0 0 20rpx rgba(0, 0, 0, 0.2);
    }

    .start {
      left: 50%;
      top: 0;
    }

    .defaultball {
      left: 50%;
      top: 0;
    }
  }
</style>
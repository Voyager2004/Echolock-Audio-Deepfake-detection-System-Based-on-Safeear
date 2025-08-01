"use strict";
const common_vendor = require("../../common/vendor.js");
const _sfc_main = {
  name: "ProgressCircleBar",
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
      validator: (value) => value >= 0 && value <= 100
    },
    // 进度条颜色（支持渐变色）
    progressColor: {
      type: [String, Array],
      default: () => ["#bced4f", "#99e252", "#72d555"]
    },
    // 内环背景色
    innerColor: {
      type: String,
      default: "transparent"
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
      return this.percent >= 5;
    },
    // 显示默认小球的条件
    showDefaultBall() {
      return this.percent > 0 && this.percent < 5;
    },
    // 外环样式
    outringStyle() {
      return {
        background: `rgba(0,0,0,0.1)`
      };
    },
    // 进度条样式
    progressBarStyle() {
      const endAngle = this.percent / 100 * 360;
      const colors = Array.isArray(this.progressColor) ? this.progressColor : [this.progressColor];
      let gradient = "conic-gradient(";
      colors.forEach((color, index) => {
        const stop = (endAngle * (index / (colors.length - 1))).toFixed(2);
        gradient += `${color} ${stop}deg${index < colors.length - 1 ? "," : ""}`;
      });
      gradient += `, transparent ${endAngle}deg)`;
      return {
        background: gradient,
        width: this.size + "rpx",
        height: this.size + "rpx"
      };
    },
    // 内环样式
    innerringStyle() {
      const innerSize = this.size * 0.7;
      return {
        width: innerSize + "rpx",
        height: innerSize + "rpx",
        backgroundColor: this.innerColor
      };
    },
    // 起始小球样式
    startBallStyle() {
      return {
        width: this.ballSize + "rpx",
        height: this.ballSize + "rpx",
        top: this.ballSize / 2 + "rpx",
        "z-index": "0",
        backgroundColor: this.progressColor[0]
      };
    },
    // 结束小球样式
    endBallStyle() {
      const angle = (this.percent / 100 * 360 - 90) * Math.PI / 180;
      const radius = (this.size - this.ballSize) / 2;
      const center = this.size / 2;
      const c = this.progressColor[this.progressColor.length - 1];
      common_vendor.index.__f__("log", "at pages/index/ring.vue:109", c);
      const x = center + radius * Math.cos(angle);
      const y = center + radius * Math.sin(angle);
      return {
        width: this.ballSize + "rpx",
        height: this.ballSize + "rpx",
        left: x + "rpx",
        top: y + "rpx",
        "z-index": "1",
        backgroundColor: c
      };
    },
    // 默认小球样式
    defaultBallStyle() {
      return {
        width: this.ballSize + "rpx",
        height: this.ballSize + "rpx",
        background: `linear-gradient(90deg, 
          ${this.progressColor[0]} 0%, 
          ${this.progressColor[this.progressColor.length - 1]} 100%)`
      };
    }
  }
};
function _sfc_render(_ctx, _cache, $props, $setup, $data, $options) {
  return common_vendor.e({
    a: $options.showProgress
  }, $options.showProgress ? {
    b: common_vendor.s($options.progressBarStyle)
  } : {}, {
    c: $options.showProgress
  }, $options.showProgress ? {
    d: common_vendor.s($options.startBallStyle)
  } : {}, {
    e: $options.showProgress
  }, $options.showProgress ? {
    f: common_vendor.s($options.endBallStyle)
  } : {}, {
    g: $options.showDefaultBall
  }, $options.showDefaultBall ? {
    h: common_vendor.s($options.defaultBallStyle)
  } : {}, {
    i: common_vendor.s($options.outringStyle),
    j: common_vendor.s($options.innerringStyle),
    k: common_vendor.t($props.percent),
    l: $props.progressColor[$props.progressColor.length - 1],
    m: $props.size + "rpx",
    n: $props.size + "rpx"
  });
}
const Component = /* @__PURE__ */ common_vendor._export_sfc(_sfc_main, [["render", _sfc_render], ["__scopeId", "data-v-c024dc92"]]);
wx.createComponent(Component);
//# sourceMappingURL=../../../.sourcemap/mp-weixin/pages/index/ring.js.map

"use strict";
const common_vendor = require("../../common/vendor.js");
if (!Array) {
  const _easycom_uni_icons2 = common_vendor.resolveComponent("uni-icons");
  const _easycom_uni_steps2 = common_vendor.resolveComponent("uni-steps");
  (_easycom_uni_icons2 + _easycom_uni_steps2)();
}
const _easycom_uni_icons = () => "../../uni_modules/uni-icons/components/uni-icons/uni-icons.js";
const _easycom_uni_steps = () => "../../uni_modules/uni-steps/components/uni-steps/uni-steps.js";
if (!Math) {
  (_easycom_uni_icons + _easycom_uni_steps)();
}
const _sfc_main = {
  __name: "record",
  setup(__props) {
    const tape = common_vendor.index.getRecorderManager();
    const audio = common_vendor.index.createInnerAudioContext();
    const state = common_vendor.ref(0);
    const time = common_vendor.ref(0);
    const tempFilePath = common_vendor.ref(null);
    const timer = common_vendor.ref(null);
    const result = common_vendor.ref(0);
    const timeDisplay = common_vendor.reactive({
      h: "00",
      m: "00",
      s: "00"
    });
    common_vendor.onLoad(() => {
      updateTime();
    });
    common_vendor.onUnmounted(() => {
      clearInterval(timer.value);
      audio.destroy();
    });
    const formatTime = (time2) => {
      return time2 < 10 ? `0${time2}` : time2;
    };
    const updateTime = () => {
      if (!timeDisplay) {
        common_vendor.index.__f__("error", "at pages/record/record.vue:112", "timeDisplay is undefined");
        return;
      }
      const total = time.value || 0;
      timeDisplay.h = formatTime(Math.floor(total / 3600));
      timeDisplay.m = formatTime(Math.floor(total % 3600 / 60));
      timeDisplay.s = formatTime(Math.floor(total % 60));
    };
    const handleRecord = () => {
      common_vendor.index.__f__("log", "at pages/record/record.vue:124", "handleRecord");
      switch (state.value) {
        case 0:
          startRecording();
          break;
        case 1:
          pauseRecording();
          break;
        case 2:
          resumeRecording();
          break;
      }
    };
    const startRecording = () => {
      state.value = 1;
      tape.start();
      startTimer();
    };
    const pauseRecording = () => {
      state.value = 2;
      clearInterval(timer.value);
      tape.pause();
    };
    const resumeRecording = () => {
      state.value = 1;
      tape.resume();
      startTimer();
    };
    const stopRecording = () => {
      state.value = 0;
      clearInterval(timer.value);
      tape.stop();
      resetTimer();
      common_vendor.index.showToast({
        title: "保存成功",
        duration: 1e3
      });
    };
    const playRecording = () => {
      if (!tempFilePath.value) {
        common_vendor.index.showModal({
          title: "提示",
          content: "请先录制音频"
        });
        return;
      }
      audio.src = tempFilePath.value;
      audio.play();
      startarti.showToast({
        title: "播放中",
        duration: 1e3
      });
    };
    const startTimer = () => {
      timer.value = setInterval(() => {
        time.value++;
        updateTime();
      }, 1e3);
    };
    const resetTimer = () => {
      time.value = 0;
      updateTime();
    };
    tape.onStop((res) => {
      tempFilePath.value = res.tempFilePath;
    });
    common_vendor.computed(() => {
      return ["开始录音", "暂停录音", "继续录音"][state.value];
    });
    const statusMessage = common_vendor.computed(() => {
      return [
        "点击红色按钮开始录音",
        "录音中...",
        "已暂停"
      ][state.value];
    });
    common_vendor.ref(false);
    const uploadFile = async () => {
      if (!tempFilePath.value)
        return;
      startProgress();
      common_vendor.index.showLoading({
        title: "上传中...",
        mask: true
      });
      try {
        common_vendor.index.__f__("log", "at pages/record/record.vue:243", tempFilePath.value);
        const res = await common_vendor.index.uploadFile({
          url: "https://whusafeear.top:3006/common/upload",
          // 替换为实际接口
          filePath: tempFilePath.value,
          name: "file",
          formData: {
            "content-type": "multipart/form-data"
          }
        });
        common_vendor.index.__f__("log", "at pages/record/record.vue:252", res);
        const receive = JSON.parse(res.data);
        common_vendor.index.__f__("log", "at pages/record/record.vue:254", receive);
        result.value = Number((parseFloat(receive.probability) * 100).toFixed(2));
        saveHistory({
          fileName: "录音文件",
          source: "录音上传",
          result: result.value,
          time: (/* @__PURE__ */ new Date()).toLocaleString()
        });
        stopProgress();
      } catch (error) {
        common_vendor.index.showModal({
          title: "上传失败",
          content: error || "fail",
          showCancel: false
        });
      } finally {
        common_vendor.index.hideLoading();
      }
    };
    const saveHistory = (data) => {
      let history = common_vendor.index.getStorageSync("detectHistory") || [];
      history.unshift(data);
      if (history.length > 50)
        history = history.slice(0, 50);
      common_vendor.index.setStorageSync("detectHistory", history);
    };
    const probabilityColor = common_vendor.computed(() => {
      const green = [0, 255, 0];
      const red = [255, 0, 0];
      const r = Math.round(red[0] * (result.value / 100) + green[0] * (1 - result.value / 100));
      const g = Math.round(red[1] * (result.value / 100) + green[1] * (1 - result.value / 100));
      const b = Math.round(red[2] * (result.value / 100) + green[2] * (1 - result.value / 100));
      const hex = (r << 16 | g << 8 | b).toString(16).padStart(6, "0");
      return `#${hex}`;
    });
    let timerForSteps = null;
    const stepOptions = [
      {
        title: "请选择音频文件"
      },
      {
        title: "音频加密上传中"
      },
      {
        title: "音频信号预处理中"
      },
      {
        title: "加载深度伪造检测模型中"
      },
      {
        title: "声学特征检测中"
      },
      {
        title: "音频检测完成"
      }
    ];
    let currentStateIndex = common_vendor.ref(0);
    const startProgress = () => {
      currentStateIndex.value = 0;
      timerForSteps = setInterval(() => {
        if (currentStateIndex.value < stepOptions.length - 2) {
          currentStateIndex.value++;
        }
      }, 3e3);
    };
    const stopProgress = () => {
      clearInterval(timerForSteps);
      currentStateIndex.value = stepOptions.length - 1;
    };
    common_vendor.onUnmounted(() => {
      clearInterval(timerForSteps);
    });
    return (_ctx, _cache) => {
      return {
        a: common_vendor.t((timeDisplay == null ? void 0 : timeDisplay.h) || "00"),
        b: common_vendor.t((timeDisplay == null ? void 0 : timeDisplay.m) || "00"),
        c: common_vendor.t((timeDisplay == null ? void 0 : timeDisplay.s) || "00"),
        d: common_vendor.p({
          type: "headphones",
          size: "30",
          color: "blue"
        }),
        e: !tempFilePath.value ? 1 : "",
        f: common_vendor.o(playRecording),
        g: state.value === 1 ? 1 : "",
        h: state.value === 2 ? 1 : "",
        i: common_vendor.o(handleRecord),
        j: common_vendor.p({
          type: "circle-filled",
          size: "30",
          color: "red"
        }),
        k: state.value === 0 ? 1 : "",
        l: common_vendor.o(stopRecording),
        m: common_vendor.t(statusMessage.value),
        n: common_vendor.p({
          type: "cloud-upload",
          size: "24",
          color: "#666"
        }),
        o: !tempFilePath.value,
        p: common_vendor.o(uploadFile),
        q: common_vendor.t(result.value),
        r: probabilityColor.value,
        s: result.value,
        t: probabilityColor.value,
        v: common_vendor.p({
          options: stepOptions,
          active: common_vendor.unref(currentStateIndex),
          ["active-color"]: "#007AFF",
          direction: "column"
        })
      };
    };
  }
};
const MiniProgramPage = /* @__PURE__ */ common_vendor._export_sfc(_sfc_main, [["__scopeId", "data-v-ef6850c5"]]);
wx.createPage(MiniProgramPage);
//# sourceMappingURL=../../../.sourcemap/mp-weixin/pages/record/record.js.map

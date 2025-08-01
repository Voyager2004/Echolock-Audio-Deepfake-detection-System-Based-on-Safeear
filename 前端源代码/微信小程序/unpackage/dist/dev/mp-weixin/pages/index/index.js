"use strict";
const common_vendor = require("../../common/vendor.js");
if (!Array) {
  const _easycom_uni_steps2 = common_vendor.resolveComponent("uni-steps");
  _easycom_uni_steps2();
}
const _easycom_uni_steps = () => "../../uni_modules/uni-steps/components/uni-steps/uni-steps.js";
if (!Math) {
  (ProgressCircleBar + _easycom_uni_steps)();
}
const ProgressCircleBar = () => "./ring.js";
const _sfc_main = {
  __name: "index",
  setup(__props) {
    const percent = common_vendor.ref(0);
    const loading = common_vendor.ref(false);
    const name = common_vendor.ref("未上传文件");
    let timer = null;
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
      timer = setInterval(() => {
        if (currentStateIndex.value < stepOptions.length - 2) {
          currentStateIndex.value++;
        }
      }, 3e3);
    };
    const stopProgress = () => {
      clearInterval(timer);
      currentStateIndex.value = stepOptions.length - 1;
    };
    common_vendor.onUnmounted(() => {
      clearInterval(timer);
    });
    const progressColor = common_vendor.computed(() => {
      const green = [0, 255, 0];
      const red = [255, 0, 0];
      const r = Math.round(red[0] * (percent.value / 100) + green[0] * (1 - percent.value / 100));
      const g = Math.round(red[1] * (percent.value / 100) + green[1] * (1 - percent.value / 100));
      const b = Math.round(red[2] * (percent.value / 100) + green[2] * (1 - percent.value / 100));
      const hex = (r << 16 | g << 8 | b).toString(16).padStart(6, "0");
      if (percent.value < 45) {
        return ["#72d555", `#${hex}`];
      } else {
        return ["#72d555", `#FFA500`, `#${hex}`];
      }
    });
    const handleUpload = async () => {
      var _a;
      try {
        const res = await common_vendor.index.chooseMessageFile({
          count: 1,
          type: "file",
          extension: ["wav", "mp3", "m4a", "flac", "aac"]
        });
        if (!((_a = res == null ? void 0 : res.tempFiles) == null ? void 0 : _a[0]))
          return;
        name.value = res.tempFiles[0].name;
        loading.value = true;
        startProgress();
        common_vendor.index.__f__("log", "at pages/index/index.vue:119", res.tempFiles[0].path);
        common_vendor.index.uploadFile({
          url: "https://whusafeear.top:3006/common/upload",
          filePath: res.tempFiles[0].path,
          name: "file",
          header: {
            // 'Transfer-Encoding':'chunked'
          },
          formData: {
            "content-type": "multipart/form-data"
          },
          success: (uploadRes) => {
            try {
              const data = JSON.parse(uploadRes.data);
              percent.value = Number((parseFloat(data.probability) * 100).toFixed(2));
              saveHistory({
                fileName: name.value,
                source: "文件上传",
                result: percent.value,
                time: (/* @__PURE__ */ new Date()).toLocaleString()
              });
            } catch (e) {
              common_vendor.index.showToast({
                title: e,
                icon: "error",
                duration: 2e3
              });
            }
          },
          fail: (err) => {
            common_vendor.index.showToast({
              title: "上传失败",
              icon: "error"
            });
            common_vendor.index.__f__("error", "at pages/index/index.vue:157", "上传失败:", err);
          },
          complete: () => {
            loading.value = false;
            stopProgress();
          }
        });
      } catch (err) {
        common_vendor.index.__f__("error", "at pages/index/index.vue:165", "文件选择错误:", err);
        common_vendor.index.showToast({
          title: "文件选择失败",
          icon: "error"
        });
        stopProgress();
      }
    };
    const saveHistory = (data) => {
      let history = common_vendor.index.getStorageSync("detectHistory") || [];
      history.unshift(data);
      if (history.length > 50)
        history = history.slice(0, 50);
      common_vendor.index.setStorageSync("detectHistory", history);
    };
    return (_ctx, _cache) => {
      return {
        a: common_vendor.p({
          size: 300,
          percent: percent.value,
          progressColor: progressColor.value,
          ["inner-color"]: "#F8F9FA",
          ["ball-size"]: 45
        }),
        b: common_vendor.t(name.value),
        c: common_vendor.t(loading.value ? "上传检测中..." : "音频文件上传"),
        d: loading.value,
        e: loading.value,
        f: common_vendor.o(handleUpload),
        g: common_vendor.p({
          options: stepOptions,
          active: common_vendor.unref(currentStateIndex),
          ["active-color"]: "#007AFF",
          direction: "column"
        })
      };
    };
  }
};
const MiniProgramPage = /* @__PURE__ */ common_vendor._export_sfc(_sfc_main, [["__scopeId", "data-v-1cf27b2a"]]);
wx.createPage(MiniProgramPage);
//# sourceMappingURL=../../../.sourcemap/mp-weixin/pages/index/index.js.map

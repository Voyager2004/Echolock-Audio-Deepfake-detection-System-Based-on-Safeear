"use strict";
const common_vendor = require("../../common/vendor.js");
if (!Array) {
  const _easycom_qiun_data_charts2 = common_vendor.resolveComponent("qiun-data-charts");
  _easycom_qiun_data_charts2();
}
const _easycom_qiun_data_charts = () => "../../uni_modules/qiun-data-charts/components/qiun-data-charts/qiun-data-charts.js";
if (!Math) {
  _easycom_qiun_data_charts();
}
const _sfc_main = {
  __name: "history",
  setup(__props) {
    const historyList = common_vendor.ref([]);
    const columnData = common_vendor.ref({
      categories: [],
      series: [{
        name: "检测结果",
        data: []
      }],
      xAxis: {
        // 隐藏 x 轴标签
        axisLabel: {
          show: false
        },
        // 隐藏 x 轴刻度线
        axisTick: {
          show: false
        },
        // 隐藏 x 轴轴线
        axisLine: {
          show: false
        }
      }
    });
    const pieData = common_vendor.ref({
      series: [{
        data: [],
        // 饼图颜色配置
        color: (params) => getColorByResult(params.value)
      }]
    });
    common_vendor.onShow(() => {
      loadHistory();
    });
    const loadHistory = () => {
      historyList.value = common_vendor.index.getStorageSync("detectHistory") || [];
      columnData.value = {
        categories: historyList.value.map((item) => " "),
        series: [{
          ...columnData.value.series[0],
          data: historyList.value.map((item) => Number(item.result))
        }]
      };
      const pieCategories = {
        "高风险": { min: 70, max: 100, count: 0 },
        "中风险": { min: 30, max: 69, count: 0 },
        "低风险": { min: 0, max: 29, count: 0 }
      };
      historyList.value.forEach((item) => {
        const result = Number(item.result);
        Object.values(pieCategories).forEach((cat) => {
          if (result >= cat.min && result <= cat.max)
            cat.count++;
        });
      });
      pieData.value = {
        series: [{
          ...pieData.value.series[0],
          data: Object.entries(pieCategories).map(([name, cat]) => ({
            name,
            value: cat.count
          }))
        }]
      };
    };
    const getColorByResult = (result) => {
      const green = [0, 255, 0];
      const red = [255, 0, 0];
      const r = Math.round(red[0] * (result / 100) + green[0] * (1 - result / 100));
      const g = Math.round(red[1] * (result / 100) + green[1] * (1 - result / 100));
      const b = Math.round(red[2] * (result / 100) + green[2] * (1 - result / 100));
      const hex = (r << 16 | g << 8 | b).toString(16).padStart(6, "0");
      return `#${hex}`;
    };
    return (_ctx, _cache) => {
      return {
        a: common_vendor.p({
          type: "column",
          chartData: columnData.value
        }),
        b: common_vendor.p({
          type: "pie",
          chartData: pieData.value
        }),
        c: common_vendor.f(historyList.value, (item, index, i0) => {
          return {
            a: common_vendor.t(item.fileName),
            b: common_vendor.t(item.result),
            c: getColorByResult(item.result),
            d: common_vendor.t(item.source),
            e: common_vendor.t(item.time),
            f: index,
            g: getColorByResult(item.result)
          };
        })
      };
    };
  }
};
const MiniProgramPage = /* @__PURE__ */ common_vendor._export_sfc(_sfc_main, [["__scopeId", "data-v-b2d018fa"]]);
wx.createPage(MiniProgramPage);
//# sourceMappingURL=../../../.sourcemap/mp-weixin/pages/history/history.js.map

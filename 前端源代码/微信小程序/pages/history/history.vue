<template>
  <view class="container">
    <view class="header">
      <text class="title">检测历史记录</text>
      <text class="subtitle">最近50条记录</text>
    </view>
    <view class="chart">
      <qiun-data-charts type="column" :chartData="columnData" />
    </view>
    <view class="chart">
      <qiun-data-charts type="pie" :chartData="pieData" />
    </view>
    <scroll-view class="history-list" scroll-y>
      <view v-for="(item, index) in historyList" :key="index" class="history-item"
        :style="{ borderLeftColor: getColorByResult(item.result) }">
        <view class="item-header">
          <text class="file-name">{{ item.fileName }}</text>
          <text class="result" :style="{ color: getColorByResult(item.result) }">
            {{ item.result }}%
          </text>
        </view>
        <view class="item-footer">
          <text class="source">{{ item.source }}</text>
          <text class="time">{{ item.time }}</text>
        </view>
      </view>
    </scroll-view>
  </view>
</template>

<script setup>
import { ref } from 'vue'
import { onShow, onReady } from '@dcloudio/uni-app'

const historyList = ref([]);

// 柱状图数据（按官方文档结构）
const columnData = ref({
  categories: [],
  series: [{
    name: "检测结果",
    data: [],
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

// 饼图数据（按官方文档结构）
const pieData = ref({
  series: [{
    data: [],
    // 饼图颜色配置
    color: (params) => getColorByResult(params.value)
  }]
});

onShow(() => {
  loadHistory();
});
const loadHistory = () => {
  historyList.value = uni.getStorageSync('detectHistory') || [];

  // 更新柱状图数据
  columnData.value = {
    categories: historyList.value.map(item => ' '),
    series: [{
      ...columnData.value.series[0],
      data: historyList.value.map(item => Number(item.result))
    }]
  };

  // 更新饼图数据（保持分类逻辑）
  const pieCategories = {
    '高风险': { min: 70, max: 100, count: 0 },
    '中风险': { min: 30, max: 69, count: 0 },
    '低风险': { min: 0, max: 29, count: 0 }
  };

  historyList.value.forEach(item => {
    const result = Number(item.result);
    Object.values(pieCategories).forEach(cat => {
      if (result >= cat.min && result <= cat.max) cat.count++;
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
  const hex = (r << 16 | g << 8 | b).toString(16).padStart(6, '0');
  return `#${hex}`;
};

</script>

<style lang="scss" scoped>
.container {
  padding: 20rpx;
  height: 100vh;
  background-color: #f5f5f5;
}

.header {
  padding: 20rpx;
  margin-bottom: 20rpx;

  .title {
    font-size: 40rpx;
    font-weight: bold;
    display: block;
  }

  .subtitle {
    font-size: 28rpx;
    color: #666;
  }
}

.history-list {
  height: calc(100vh - 120rpx);
}

.history-item {
  background: #fff;
  border-radius: 12rpx;
  padding: 20rpx;
  margin-bottom: 20rpx;
  border-left: 8rpx solid;

  .item-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10rpx;

    .file-name {
      font-size: 32rpx;
      max-width: 70%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .result {
      font-size: 32rpx;
      font-weight: bold;
    }
  }

  .item-footer {
    display: flex;
    justify-content: space-between;
    font-size: 24rpx;
    color: #999;
  }
}

.chart {
  width: 100%;
  height: 400rpx;
  padding: 0;
}
</style>
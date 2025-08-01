import { window } from "@kit.ArkUI"

export class WindowManager {
  // 开启沉浸式显示
  static async enableFullScreen() {
    const context = getContext()
    const win = await window.getLastWindow(context)
    win.setWindowLayoutFullScreen(true)

    const topHeight = win.getWindowAvoidArea(window.AvoidAreaType.TYPE_SYSTEM)
    AppStorage.setOrCreate("topHeight", px2vp(topHeight.topRect.height))
  }

  //  关闭沉浸式显示
  static async disableFullScreen() {
    const context = getContext()
    const win = await window.getLastWindow(context)
    win.setWindowLayoutFullScreen(false)

    AppStorage.setOrCreate("topHeight", 0)
  }

  // 设置安全区域文字颜色为浅色
  static async setStatusBarLight() {
    const context = getContext()
    const win = await window.getLastWindow(context)
    win.setWindowSystemBarProperties({ statusBarContentColor: "#ffffff" })
  }

  // 设置安全区域文字颜色为深色
  static async setStatusBarDark() {
    const context = getContext()
    const win = await window.getLastWindow(context)
    win.setWindowSystemBarProperties({ statusBarContentColor: "#000000" })
  }
}
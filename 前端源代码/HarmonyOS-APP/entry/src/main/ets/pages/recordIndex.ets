import { WindowManager } from '../utils/WindowManager'
import { Permissions } from '@kit.AbilityKit'
import { promptAction, router } from '@kit.ArkUI'
import { Permission } from '../utils/Permission'
import { audioDB, IInterviewAudioItem } from '../Audio/AudioDB'
import { AudioPlayer, AudioRecordComp, RecordingListItemComp } from '../Audio/index'

@Entry
@Component
struct Index {
  @StorageProp('topHeight')
  topHeight: number = 0
  @State list: IInterviewAudioItem[] = []
  @State currentItem: IInterviewAudioItem = {} as IInterviewAudioItem
  permissions: Permissions[] = ['ohos.permission.MICROPHONE']
  confirmConfig: promptAction.ShowDialogOptions = {
    title: "温馨提示",
    message: "未授权使用麦克风将无法使用该面试录音功能，是否前往设置进行授权？",
    buttons: [
      { text: '离开', color: '#ccc' },
      { text: '去授权', color: '#ccc' }
    ]
  }

  async aboutToAppear() {
    WindowManager.enableFullScreen()
    this.getPermission()
    await audioDB.createStore()
    this.list = audioDB.getAllData()
  }

  /** 请求权限 */
  async getPermission() {
    const isAgree = await Permission.requestPermissions(this.permissions)
    if (isAgree) {
      return
    }
    const res2 = await promptAction.showDialog(this.confirmConfig)
    if (res2.index === 0) {
      return
    }
    const isAgreeAgain = await Permission.openPermissionsSetting(this.permissions)
    if (isAgreeAgain) {
      return
    }
  }

  build() {
    /** 整体页面 **/
    RelativeContainer() {
      /** 导航栏 **/
      Row() {
        Image($r('app.media.ic_public_arrow_left'))
          .width(30)
          .onClick(()=>{
            router.back()
          })

      }
      .padding({ left: 10, right: 10 })
      .justifyContent(FlexAlign.SpaceBetween)
      .height(40)
      .width('100%')
      .alignRules({
        left: { anchor: "__container__", align: HorizontalAlign.Start },
        top: { anchor: "__container__", align: VerticalAlign.Top },
      })
      .id("navBar")

      /** 标题 **/
      Text("所有录音")
        .fontSize(32)
        .fontColor('#ccc')
        .margin({ left: 10, top: 10 })
        .fontWeight(700)
        .alignRules({
          left: { anchor: "__container__", align: HorizontalAlign.Start },
          top: { anchor: "navBar", align: VerticalAlign.Bottom },
        })
        .id("title")

      /** 录音列表 **/
      List() {
        ForEach(this.list, (item: IInterviewAudioItem, index) => {
          ListItem() {
            Column() {
              RecordingListItemComp({
                item: item,
              })
              if (this.currentItem.id === item.id) {
                AudioPlayer({
                  item: this.currentItem,
                  getList: () => {
                    this.list = audioDB.getAllData()
                  }
                })
              }
            }
          }
          .onClick(() => {
            if (this.currentItem.id === item.id) {
              this.currentItem = {} as IInterviewAudioItem
              return
            }
            this.currentItem = item
          })
        })
      }
      .padding({ top: 20 })
      .divider({
        strokeWidth: 2,
        color:'#ccc'
      })
      .alignRules({
        top: { anchor: "title", align: VerticalAlign.Bottom },
        bottom: { anchor: "recorder", align: VerticalAlign.Top }
      })
      .id("records")

      /** 录音按钮 **/
      AudioRecordComp({
        onEnd: (item: IInterviewAudioItem) => {
          this.list = audioDB.add(item!)
        },
        listLength: this.list.length
      })
    }
    .height('100%')
    .padding({ top: this.topHeight, bottom: 15 })
  }
}


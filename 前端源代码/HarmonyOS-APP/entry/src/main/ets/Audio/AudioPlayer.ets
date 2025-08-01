import { audioDB, IInterviewAudioItem } from '.'
import { media } from '@kit.MediaKit'
import { fileIo } from '@kit.CoreFileKit'
import { InputDialog } from '../commons'

@Component
export struct AudioPlayer {
  @State
  playing: boolean = false
  @Prop item: IInterviewAudioItem = {} as IInterviewAudioItem
  avPlayer?: media.AVPlayer
  @State total: number = 0
  @State value: number = 0
  getList = () => {
  }
  dialog: CustomDialogController = new CustomDialogController({
    builder: InputDialog({
      name: this.item.name,
      onSubmit: (name) => {
        this.item.name = name
        audioDB.updateData(this.item)
        this.getList()
        this.dialog.close()
      }
    }),
    customStyle: true
  })

  async aboutToAppear() {
    await this.startPlay()
  }

  /** 计算持续时间 **/
  calculateDuration(time: number) {
    const totalSecond = Math.floor(time / 1000)
    const mm = Math.floor(totalSecond / 60)
    const ss = totalSecond - mm * 60
    return `${mm < 10 ? "0" + mm.toString() : mm}:${ss < 10 ? "0" + ss.toString() : ss}`
  }

  /** 开始播放 **/
  async startPlay() {
    try {
      const file = fileIo.openSync(this.item.path, fileIo.OpenMode.READ_ONLY)
      const avPlayer = await media.createAVPlayer()
      avPlayer.on('stateChange', state => {
        if (state === 'initialized') {
          avPlayer.prepare()
        } else if (state === 'prepared') {
          this.total = avPlayer.duration
          avPlayer.play()
        }
      })
      // 当前播放时间改变
      avPlayer.on('timeUpdate', (time) => {
        this.value = time
      })
      avPlayer.url = `fd://${file.fd}`
      this.avPlayer = avPlayer
      this.playing = true
    } catch (e) {
      console.error('startPlay', JSON.stringify(e))
    }
  }

  /** 停止播放 **/
  stopPlay() {
    if (this.avPlayer) {
      this.avPlayer.stop()
      this.avPlayer.release()
      this.playing = false
    }
  }

  build() {
    RelativeContainer() {

      Progress({ value: this.value, total: this.total, type: ProgressType.Linear })
        .layoutWeight(1)
        .margin({ top: 10 })
        .backgroundColor(Color.Gray)
        .id('progress')

      Text(`${this.calculateDuration(this.value)}`)
        .fontColor(Color.Gray)
        .fontSize(18)
        .margin({ top: 5 })
        .alignRules({
          top: { anchor: "progress", align: VerticalAlign.Bottom }
        })
        .id('value')
      Text(`-${this.calculateDuration(this.total - this.value)}`)
        .fontColor(Color.Gray)
        .fontSize(18)
        .alignRules({
          top: { anchor: "progress", align: VerticalAlign.Bottom },
          right: { anchor: "__container__", align: HorizontalAlign.End }
        })
        .id('remainTime')

      Image(!this.playing || this.value === this.total ? $r('sys.media.ohos_ic_public_play') :
      $r('sys.media.ohos_ic_public_pause'))
        .width(30)
        .aspectRatio(1)
        .fillColor(Color.Gray)
        .onClick(() => {
          if (!this.playing) {
            this.startPlay()
          } else {
            this.stopPlay()
          }
        })
        .alignRules({
          middle: { anchor: "__container__", align: HorizontalAlign.Center },
          bottom: { anchor: "__container__", align: VerticalAlign.Bottom }
        })
        .id("playIcon")

      Image($r('app.media.ic_public_delete'))
        .width(24)
        .fillColor(Color.Green)
        .onClick(() => {
          audioDB.deleteData(this.item.id!)
          this.getList()
        })
        .alignRules({
          right: { anchor: "__container__", align: HorizontalAlign.End },
          center: { anchor: "playIcon", align: VerticalAlign.Center }
        })
        .id('delete')

      Image($r('app.media.ic_public_edit'))
        .width(24)
        .fillColor(Color.Green)
        .onClick(() => {
          this.dialog.open()
          this.stopPlay()
        })
        .alignRules({
          left: { anchor: "__container__", align: HorizontalAlign.Start },
          center: { anchor: "playIcon", align: VerticalAlign.Center }
        })
        .id('edit')
    }
    .width('90%')
    .height(100)
    .padding({ bottom: 20 })
  }
}
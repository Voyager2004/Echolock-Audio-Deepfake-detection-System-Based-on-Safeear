import { AudioPlayer, IInterviewAudioItem } from "."

@Component
export struct RecordingListItemComp {
  @Prop
  item: IInterviewAudioItem = {} as IInterviewAudioItem
  duration: string = ''

  /** 计算持续时间 **/
  calculateDuration() {
    const totalSecond = Math.floor(this.item.duration / 1000)
    const mm = Math.floor(totalSecond / 60)
    const ss = totalSecond - mm * 60
    this.duration = `${mm < 10 ? "0" + mm.toString() : mm}:${ss < 10 ? "0" + ss.toString() : ss}`
  }

  aboutToAppear(): void {
    this.calculateDuration()
  }

  build() {
    RelativeContainer() {
      Text(this.item.name)
        .fontWeight(600)
        .fontSize(22)
        .fontColor(Color.Gray)
        .padding({ top: 10 })
        .id("text")
      Text(this.item.create_time.toString())
        .fontColor(Color.Gray)
        .padding({ top: 10 })
        .fontSize(18)
        .alignRules({
          top: { anchor: "text", align: VerticalAlign.Bottom }
        })
        .id("day")
      Text(this.duration.toString())
        .fontColor(Color.Gray)
        .fontSize(18)
        .alignRules({
          bottom: { anchor: "day", align: VerticalAlign.Bottom },
          right: { anchor: "__container__", align: HorizontalAlign.End }
        })
        .id("time")
    }
    .width('100%')
    .height(80)
    .padding(10)
  }
}
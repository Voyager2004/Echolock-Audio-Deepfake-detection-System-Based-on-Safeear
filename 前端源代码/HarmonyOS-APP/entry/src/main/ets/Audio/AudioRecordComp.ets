import { AudioBoComp, IInterviewAudioItem } from '.'
import { media } from '@kit.MediaKit'
import { fileIo ,Environment } from '@kit.CoreFileKit'
import fs from '@ohos.file.fs'
import { PromptAction, promptAction } from '@kit.ArkUI'

@Component
export struct AudioRecordComp {
  @State maxAmplitude: number = 0
  @State isRecording: boolean = false
  @Prop listLength: number
  @State date:number=0
  avRecorder?: media.AVRecorder
  fd?: number
  filePath?: string
  timer?: number
  startTime = 0
  onEnd = (item: IInterviewAudioItem) => {
  }

  /** 开始录音 **/
  async startRecord() {
    this.isRecording = true
    // 1. 准备一个文件接收录音
    const ctx = getContext(this)
    this.date=Date.now()
    const filePath = ctx.filesDir + '/' + this.date + '.m4a'
    this.filePath = filePath
    const file = fileIo.openSync(filePath, fileIo.OpenMode.CREATE | fileIo.OpenMode.READ_WRITE)
    this.fd = file.fd
    // 2. 准备路由配置对象
    const config: media.AVRecorderConfig = {
      audioSourceType: media.AudioSourceType.AUDIO_SOURCE_TYPE_MIC,
      profile: {
        audioBitrate: 100000, // 音频比特率
        audioChannels: 1, // 音频声道数
        audioCodec: media.CodecMimeType.AUDIO_AAC, // 音频编码格式
        audioSampleRate: 48000, // 音频采样率
        fileFormat: media.ContainerFormatType.CFT_MPEG_4A, // 封装格式
      },
      url: `fd://${file.fd}`
    }
    // 3. 开始录制
    const avRecorder = await media.createAVRecorder()
    await avRecorder.prepare(config)
    await avRecorder.start()
    this.startTime = Date.now()
    this.avRecorder = avRecorder
    // 4. 每100ms获取一下声音振幅
    this.timer = setInterval(async () => {
      this.maxAmplitude = await avRecorder.getAudioCapturerMaxAmplitude()
    }, 100)
  }

  /** 停止录音 **/
  async stopRecord() {
    if (this.avRecorder) {
      this.isRecording = false
      clearInterval(this.timer)
      await this.avRecorder.stop()
      await this.avRecorder.release()
      fileIo.closeSync(this.fd)
      this.maxAmplitude = 0

      const fileInfo = fileIo.statSync(this.filePath)
      this.onEnd({
        id: null,
        name: `${this.date}`,
        path: this.filePath!,
        duration: Date.now() - this.startTime,
        size: fileInfo.size,
        create_time: new Date().toLocaleString()
      })
      // 在录音结束后将文件复制到指定路径
      this.copyFile();
    }
    }
  async copyFile() {
  try {
    const downloadPath = Environment.getUserDownloadDir();

    let sanFile = fs.openSync(this.filePath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE)

    let pubFile = fs.openSync(downloadPath+'/'+this.date +'.m4a', fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE)

    // 将文件从沙箱路拷贝到公共路径

    fs.copyFileSync(sanFile.fd, pubFile.fd)
    promptAction.showToast({message:'已保存至'+downloadPath+'/'+this.date +'.m4a'})
  } catch (error) {
    promptAction.showToast(error);
  }
}
  // 新增方法：复制文件到指定路径

  build() {
    Row() {
      Text()
        .width(60)
        .aspectRatio(1)
        .backgroundColor(Color.Red)
        .border({ width: 5, color: Color.White, radius: 30 })
        .onClick(async () => {
          await this.startRecord()
        })
        .bindSheet($$this.isRecording, this.recordBindBuilder(), {
          height: SheetSize.FIT_CONTENT,
          showClose: false,
        })
    }
    .justifyContent(FlexAlign.Center)
    .backgroundColor("#eee")
    .width('100%')
    .height(100)
    .alignRules({
      left: { anchor: "__container__", align: HorizontalAlign.Start },
      bottom: { anchor: "__container__", align: VerticalAlign.Bottom }
    })
    .id("recorder")
  }

  /** 开始录音半模态弹窗 **/
  @Builder
  recordBindBuilder() {
    RelativeContainer() {
      Text("新录音")
        .fontWeight(600)
        .fontSize(24)
        .fontColor(Color.Gray)
        .padding({ top: 30 })
        .alignRules({
          middle: { anchor: "__container__", align: HorizontalAlign.Center },
        })
        .id("recorderTitle")
      AudioBoComp({
        maxAmplitude: this.maxAmplitude
      })
        .alignRules({
          middle: { anchor: "__container__", align: HorizontalAlign.Center },
          center: { anchor: "__container__", align: VerticalAlign.Center }
        })
        .id("audioBo")
      Text()
        .width(60)
        .aspectRatio(1)
        .border({ width: 5, color: Color.White, radius: 30 })
        .margin({ bottom: 30 })
        .alignRules({
          bottom: { anchor: "__container__", align: VerticalAlign.Bottom },
          middle: { anchor: "__container__", align: HorizontalAlign.Center },
        })
        .id("recorderBorder")
      Text()
        .width(25)
        .height(25)
        .backgroundColor(Color.Red)
        .borderRadius(5)
        .alignRules({
          center: { anchor: "recorderBorder", align: VerticalAlign.Center },
          middle: { anchor: "recorderBorder", align: HorizontalAlign.Center },
        })
        .id("recorderButton")
        .onClick(async () => {
          await this.stopRecord()
        })
    }
    .height(300)
    .backgroundColor("#eee")
  }
}



import router from '@ohos.router'
import { rcp } from '@kit.RemoteCommunicationKit';
import { picker } from '@kit.CoreFileKit';
import { common } from '@kit.AbilityKit';
import fs from '@ohos.file.fs'
import { BusinessError } from '@kit.BasicServicesKit';
import { BottomOffset, promptAction } from '@kit.ArkUI';




@Entry
@Component
struct FakeDetectionPage {
  @State percent: number = 0 // 伪造概率百分比
  @State loading: boolean = false // 上传状态
  @State name:string ='未上传文件'

  build() {
    Column() {
      // 概率显示区域
      Text(`${this.percent}%`)
        .fontSize(80)
        .fontWeight(FontWeight.Bold)
        .fontColor(this.percent > 50 ? Color.Red : Color.Green)
        .margin({ top: '30%' })

      // 概率描述文本
      Text(`音频伪造概率`)
        .fontSize(20)
        .margin({ top: '10%',bottom:'10%' })
        .opacity(0.8)

      // 文件描述文本
      Text(`${this.name}`)
        .fontSize(16)
        .margin({ top: '10%',bottom:'10%' })
        .opacity(0.6)
        .width('80%')
        .textAlign(TextAlign.Center)

      // 上传按钮
      Button(this.loading?"上传检测中":"音频文件上传")
        .enabled(!this.loading)
        .onClick(async () => {
          let context = getContext(this) as common.Context; // 请确保 getContext(this) 返回结果为 UIAbilityContext
          // 创建文件选择器实例
          const documentViewPicker = new picker.DocumentViewPicker(context);
          let documentSelectOptions = new picker.DocumentSelectOptions();
          documentSelectOptions.maxSelectNumber = 1
          documentSelectOptions.fileSuffixFilters = ['.wav','.mp3','m4a'];
          //选择是否对指定文件或目录授权，true为授权，当为true时，defaultFilePathUri为必选参数，拉起文管授权界面；false为非授权，拉起常规文管界面（可选）
          documentSelectOptions.authMode = true;
          await documentViewPicker.select(documentSelectOptions).then((documentSelectResult: Array<string>) => {

            const session = rcp.createSession({
              headers: {
                "content-type": 'multipart/form-data'
              }
            });
            this.name=documentSelectResult[0]
            if (this.name){this.loading = true}
            // 文件上传的时候一定要先拷贝一份，然后在进行上传，要不然没权限操作文件
            const context = getContext(this)
            const fileType = this.name.substring(this.name.lastIndexOf(".") + 1);
            // （以时间戳）生成一个新的文件名
            const fileName = Date.now() + '.' + fileType
            // 通过缓存路径+文件名 拼接出完整的路径
            const copyFilePath = context.cacheDir + '/' + fileName
            // promptAction.showToast({
            //   message: copyFilePath,
            //   duration: 5000,
            //   bottom: 50
            // });
            // 将文件 拷贝到 临时目录
            const file = fs.openSync(documentSelectResult[0], fs.OpenMode.READ_ONLY)
            fs.copyFileSync(file.fd, copyFilePath)

            const formFieldFileValue: rcp.FormFieldFileValue = {
              contentType: "audio/wav",
              remoteFileName: fileName,
              contentOrPath: copyFilePath,
            };

            const formData = new rcp.MultipartForm({
              'file': formFieldFileValue // file表示服务器端接收的数据的属性名
            })
            let uploadFromFile: rcp.UploadFromFile = {
              fileOrPath: copyFilePath,
            }

            // 参数一就是服务器端接口地址
            session.post('http://110.41.61.229:3006/common/upload', formData).then(response => {
              // promptAction.showToast({
              //   message: `Reponse succeed: ${response}`,
              //   duration: 5000
              // });
              interface result {
                message: string,
                modelLogits: string,
                probability: string,
                predictedLabel:string
              }

              const a:result=response.toJSON() as result
              this.percent=Number((parseFloat(a.probability)*100).toFixed(2))
              promptAction.showToast({
                message: a.message,
                duration:2000,
                bottom: 50
              })
            }
            ).catch((err: BusinessError) => {
              promptAction.showToast({
                message: err.message,
                bottom: 50
              })
            }).then(() => {
              this.loading = false
            })

          //   session.uploadFromFile('http://110.41.61.229:3006/common/upload', uploadFromFile)
          //     .then((res: rcp.Response) =>{
          //         promptAction.showToast({
          //           message: `Reponse succeed: ${res}`,
          //           duration: 5000
          //         });
          //     })

          })

          // await setTimeout(() => {
          //   // 生成随机概率（0-100）
          //   const fakeProbability = Math.floor(Math.random() * 101);
          //   this.percent = fakeProbability;
          //
          //   // promptAction.showToast({
          //   //   message: `检测完成：伪造概率${fakeProbability}%`
          //   // });
          //   this.loading = false
          // }, 1000); // 模拟上传检测时间

        })
        .width('60%')
        .height('8%')

      Button("去录音")
        .width("60%")
        .height('8%')
        .margin({ top: "10%" })
        .onClick(()=>{
          router.pushUrl({
            url: "pages/recordIndex",
          }, router.RouterMode.Single)
        })



    }.width('100%')


  }

}

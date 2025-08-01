@Component
export struct AudioBoComp {
  @Watch('onChange')
  @Prop maxAmplitude: number
  @State per: number = 0

  onChange() {
    animateTo({ duration: 100 }, () => {
      if (this.maxAmplitude < 500) {
        this.per = 0
      } else if (this.maxAmplitude > 30000) {
        this.per = 1
      } else {
        this.per = this.maxAmplitude / 30000
      }
    })
  }

  build() {
    Row({ space: 5 }) {
      ForEach(Array.from({ length: 30 }), (item: number, index) => {

        if (index % 2 === 0) {
          Column()
            .layoutWeight(1)
            .height(this.per * 100 * Math.random())
            .backgroundColor(Color.Red)
        } else {
          Column()
            .layoutWeight(1)
            .height(this.per * 100 * Math.random())
            .backgroundColor(Color.Red)
        }

      })
    }
    .width('100%')
    .height(100)
  }
}
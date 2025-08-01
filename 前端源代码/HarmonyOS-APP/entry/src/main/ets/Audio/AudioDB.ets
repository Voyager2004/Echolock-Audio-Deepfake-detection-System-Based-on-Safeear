import { relationalStore, ValuesBucket } from '@kit.ArkData'

export interface IInterviewAudioItem extends ValuesBucket {
  id: number | null
  name: string
  path: string
  duration: number
  size: number
  create_time: string
}

export class AudioDB {
  store?: relationalStore.RdbStore
  tableName = 'audio'

  /** 创建数据库 **/
  async createStore() {
    this.store = await relationalStore.getRdbStore(getContext(), {
      name: this.tableName + '.db',
      securityLevel: relationalStore.SecurityLevel.S1
    })
    this.store?.executeSql(`
    CREATE TABLE IF NOT EXISTS ${this.tableName} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          path TEXT NOT NULL,
          duration INTEGER NOT NULL,
          size INTEGER NOT NULL,
          create_time TEXT NOT NULL
        )
    `)
  }

  /** 向表中插入一条数据 **/
  add(newItem: IInterviewAudioItem): IInterviewAudioItem[] {
    const res = this.store?.insertSync(this.tableName, newItem)
    if (!res) {
      AlertDialog.show({ message: '数据新增失败' })
    }
    return this.getAllData()
  }

  /** 获取行数 **/
  getNum() {
    const pre = new relationalStore.RdbPredicates(this.tableName)
    const res = this.store?.querySync(pre)
    AlertDialog.show({ message: this.tableName + "中，一共有" + res?.rowCount.toString() + "条数据" })
    res?.close()
  }

  /** 获取所有数据 **/
  getAllData() {
    const pre = new relationalStore.RdbPredicates(this.tableName)
    const res = this.store?.querySync(pre)
    const list = [] as IInterviewAudioItem[]
    while (res?.goToNextRow()) {
      const row = res?.getRow() as IInterviewAudioItem
      list.push(row)
    }
    res?.close()
    return list
  }

  /** 修改数据 **/
  updateData(newItem: IInterviewAudioItem): IInterviewAudioItem[] {
    const pre = new relationalStore.RdbPredicates(this.tableName)
    pre.equalTo('id', newItem.id)
    const res = this.store?.updateSync(newItem, pre)
    if (!res) {
      AlertDialog.show({ message: '数据更新失败' })
    }
    return this.getAllData()
  }

  /** 删除数据 **/
  deleteData(id: number): IInterviewAudioItem[] {
    const pre = new relationalStore.RdbPredicates(this.tableName)
    pre.equalTo('id', id)
    const res = this.store?.deleteSync(pre)
    if (!res) {
      AlertDialog.show({ message: '数据删除失败' })
    }
    return this.getAllData()
  }
}


export const audioDB = new AudioDB()
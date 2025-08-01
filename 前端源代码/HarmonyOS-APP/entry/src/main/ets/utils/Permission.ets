import { abilityAccessCtrl, Permissions } from '@kit.AbilityKit'

export class Permission {
  static async requestPermissions(permissions: Permissions[]) {
    const mgr = abilityAccessCtrl.createAtManager()
    const ctx = getContext()
    const res = await mgr.requestPermissionsFromUser(ctx, permissions)
    const isEvery = await res.authResults.every((item) => {
      return item === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED
    })
    return isEvery
  }

  static async openPermissionsSetting(permissions: Permissions[]) {
    const mgr = abilityAccessCtrl.createAtManager()
    const ctx = getContext()
    const res3 = await mgr.requestPermissionOnSetting(ctx, permissions)
    const isEvery = res3.every(item => item === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED)
    return isEvery
  }
}
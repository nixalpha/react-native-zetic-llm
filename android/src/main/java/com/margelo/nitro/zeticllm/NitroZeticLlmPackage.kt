package com.margelo.nitro.zeticllm

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class NitroZeticLlmPackage : ReactPackage {
  override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
    ZeticLLMContextHolder.setContext(reactContext)
    return emptyList()
  }

  override fun createViewManagers(
    reactContext: ReactApplicationContext
  ): List<ViewManager<*, *>> {
    ZeticLLMContextHolder.setContext(reactContext)
    return emptyList()
  }

  companion object {
    init {
      NitroZeticLlmOnLoad.initializeNative()
    }
  }
}

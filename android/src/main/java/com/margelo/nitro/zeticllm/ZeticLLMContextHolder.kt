package com.margelo.nitro.zeticllm

import android.content.Context
import com.facebook.react.bridge.ReactApplicationContext

internal object ZeticLLMContextHolder {
  @Volatile
  private var context: Context? = null
  @Volatile
  private var reactContext: ReactApplicationContext? = null

  fun setContext(value: Context) {
    context = value.applicationContext
    if (value is ReactApplicationContext) {
      reactContext = value
    }
  }

  fun requireContext(): Context {
    return context
      ?: throw IllegalStateException(
        "React application context is not ready. Ensure NitroZeticLlmPackage is autolinked before loading a model."
      )
  }

  fun requireReactContext(): ReactApplicationContext {
    return reactContext
      ?: throw IllegalStateException(
        "React application context is not ready. Ensure NitroZeticLlmPackage is autolinked before starting the agent."
      )
  }
}

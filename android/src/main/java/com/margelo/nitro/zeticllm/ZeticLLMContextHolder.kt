package com.margelo.nitro.zeticllm

import android.content.Context

internal object ZeticLLMContextHolder {
  @Volatile
  private var context: Context? = null

  fun setContext(value: Context) {
    context = value.applicationContext
  }

  fun requireContext(): Context {
    return context
      ?: throw IllegalStateException(
        "React application context is not ready. Ensure NitroZeticLlmPackage is autolinked before loading a model."
      )
  }
}

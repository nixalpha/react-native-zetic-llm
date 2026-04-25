package com.margelo.nitro.zeticllm

import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel

@DoNotStrip
class HybridZeticLLMModel(
  private var model: ZeticMLangeLLMModel?
) : HybridZeticLLMModelSpec() {
  private val lock = Any()
  private var isGenerating = false

  override fun generate(
    prompt: String,
    onToken: ((TokenEvent) -> Unit)?
  ): Promise<GenerateResult> {
    return Promise.async {
      val activeModel = beginGeneration()
      try {
        val runResult = activeModel.run(prompt)
        val output = StringBuilder()
        var generatedTokens = 0
        var finalStatus = runResult.status

        while (true) {
          val next = activeModel.waitForNextToken()
          finalStatus = next.status
          val count = next.generatedTokens
          if (count == 0) {
            break
          }

          generatedTokens = count
          output.append(next.token)
          onToken?.invoke(
            TokenEvent(
              token = next.token,
              generatedTokens = count.toDouble(),
              status = next.status.toDouble()
            )
          )
        }

        GenerateResult(
          text = output.toString(),
          promptTokens = runResult.promptTokens.toDouble(),
          generatedTokens = generatedTokens.toDouble(),
          status = finalStatus.toDouble()
        )
      } finally {
        endGeneration()
      }
    }
  }

  override fun cleanUp(): Promise<Unit> {
    return Promise.async {
      val activeModel = synchronized(lock) { model }
        ?: throw IllegalStateException("MODEL_RELEASED: This model has already been released.")
      activeModel.cleanUp()
    }
  }

  override fun release() {
    val releasedModel = synchronized(lock) {
      val current = model
      model = null
      isGenerating = false
      current
    }
    releasedModel?.deinit()
  }

  private fun beginGeneration(): ZeticMLangeLLMModel {
    return synchronized(lock) {
      val activeModel = model
        ?: throw IllegalStateException("MODEL_RELEASED: This model has already been released.")
      if (isGenerating) {
        throw IllegalStateException("GENERATION_IN_PROGRESS: This model is already generating.")
      }
      isGenerating = true
      activeModel
    }
  }

  private fun endGeneration() {
    synchronized(lock) {
      isGenerating = false
    }
  }
}

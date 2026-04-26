package com.margelo.nitro.zeticllm

import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import com.zeticai.mlange.*
import com.zeticai.mlange.core.cache.ModelCacheHandlingPolicy
import com.zeticai.mlange.core.model.APType
import com.zeticai.mlange.core.model.ModelLoadingStatus
import com.zeticai.mlange.core.model.llm.*

private const val BYTES_PER_GIGABYTE = 1_000_000_000.0

@DoNotStrip
class HybridZeticLLM : HybridZeticLLMSpec() {
  override fun loadModel(
    config: NativeLoadModelConfig,
    onDownload: ((Double) -> Unit)?
  ): Promise<HybridZeticLLMModelSpec> {
    return Promise.async {
      val context = ZeticLLMContextHolder.requireContext()
      val version = config.version?.toInt()
      val initOption = makeInitOption(config.initOption)
      val cachePolicy = makeCachePolicy(config.cacheHandlingPolicy)
      val downloadedGigabytes = makeDownloadedGigabytesCallback(onDownload)
      val statusChanged = makeStatusChangedCallback()

      Log.d(
        TAG,
        "loadModel: name=${config.name}, version=${version ?: "latest"}, mode=${config.modelMode ?: "RUN_AUTO"}, explicitRuntime=${config.explicitRuntime != null}"
      )

      val model = MLangeDownloadSize.withDownloadedBytes(
        context = context,
        onBytes = downloadedGigabytes
      ) {
        config.explicitRuntime?.let { explicit ->
          ZeticMLangeLLMModel(
            context = context,
            personalKey = config.personalKey,
            name = config.name,
            version = version,
            target = makeTarget(explicit.target),
            quantType = makeQuantType(explicit.quantType),
            apType = makeAPType(explicit.apType),
            onProgress = null,
            onStatusChanged = statusChanged,
            cacheHandlingPolicy = cachePolicy,
            initOption = initOption
          )
        } ?: ZeticMLangeLLMModel(
          context = context,
          personalKey = config.personalKey,
          name = config.name,
          version = version,
          modelMode = makeModelMode(config.modelMode),
          dataSetType = makeDataSetType(config.dataSetType),
          onProgress = null,
          onStatusChanged = statusChanged,
          cacheHandlingPolicy = cachePolicy,
          initOption = initOption
        )
      }

      Log.d(TAG, "loadModel: model initialized for ${config.name}")
      HybridZeticLLMModel(model)
    }
  }

  private fun normalize(value: String?): String = value?.trim()?.uppercase() ?: ""

  private fun makeStatusChangedCallback(): (ModelLoadingStatus) -> Unit =
    { status -> Log.d(TAG, "model loading status: $status") }

  private fun makeDownloadedGigabytesCallback(onDownload: ((Double) -> Unit)?): (Long) -> Unit =
    { bytes ->
      val gigabytes = bytes.toDouble() / BYTES_PER_GIGABYTE
      Log.d(TAG, "downloaded model size=${formatGigabytes(gigabytes)} GB")
      onDownload?.invoke(gigabytes)
    }

  private fun formatGigabytes(gigabytes: Double): String =
    "%.3f".format(gigabytes)

  private fun makeModelMode(value: String?): LLMModelMode =
    when (normalize(value)) {
      "RUN_SPEED" -> LLMModelMode.RUN_SPEED
      "RUN_ACCURACY" -> LLMModelMode.RUN_ACCURACY
      else -> LLMModelMode.RUN_AUTO
    }

  private fun makeDataSetType(value: String?): LLMDataSetType? =
    when (normalize(value)) {
      "" -> null
      "MMLU" -> LLMDataSetType.MMLU
      "TRUTHFULQA" -> LLMDataSetType.TRUTHFULQA
      "CNN_DAILYMAIL" -> LLMDataSetType.CNN_DAILYMAIL
      "GSM8K" -> LLMDataSetType.GSM8K
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported dataSetType: $value")
    }

  private fun makeCachePolicy(value: String?): ModelCacheHandlingPolicy =
    when (normalize(value)) {
      "KEEP_EXISTING" -> ModelCacheHandlingPolicy.KEEP_EXISTING
      else -> ModelCacheHandlingPolicy.REMOVE_OVERLAPPING
    }

  private fun makeKVPolicy(value: String?): LLMKVCacheCleanupPolicy =
    when (normalize(value)) {
      "DO_NOT_CLEAN_UP" -> LLMKVCacheCleanupPolicy.DO_NOT_CLEAN_UP
      else -> LLMKVCacheCleanupPolicy.CLEAN_UP_ON_FULL
    }

  private fun makeInitOption(option: NativeLLMInitOption?): LLMInitOption =
    LLMInitOption(
      kvCacheCleanupPolicy = makeKVPolicy(option?.kvCacheCleanupPolicy),
      nCtx = option?.nCtx?.toInt() ?: 2048
    )

  private fun makeTarget(value: String): LLMTarget =
    when (normalize(value)) {
      "LLAMA_CPP" -> LLMTarget.LLAMA_CPP
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported target: $value")
    }

  private fun makeQuantType(value: String): LLMQuantType =
    when (normalize(value)) {
      "GGUF_QUANT_ORG" -> LLMQuantType.GGUF_QUANT_ORG
      "GGUF_QUANT_F16" -> LLMQuantType.GGUF_QUANT_F16
      "GGUF_QUANT_BF16" -> LLMQuantType.GGUF_QUANT_BF16
      "GGUF_QUANT_Q8_0" -> LLMQuantType.GGUF_QUANT_Q8_0
      "GGUF_QUANT_Q6_K" -> LLMQuantType.GGUF_QUANT_Q6_K
      "GGUF_QUANT_Q4_K_M" -> LLMQuantType.GGUF_QUANT_Q4_K_M
      "GGUF_QUANT_Q3_K_M" -> LLMQuantType.GGUF_QUANT_Q3_K_M
      "GGUF_QUANT_Q2_K" -> LLMQuantType.GGUF_QUANT_Q2_K
      "GGUF_QUANT_NUM_TYPES" -> LLMQuantType.GGUF_QUANT_NUM_TYPES
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported quantType: $value")
    }

  private fun makeAPType(value: String?): APType =
    when (normalize(value)) {
      "", "CPU" -> APType.CPU
      "GPU" -> APType.GPU
      "NPU" -> APType.NPU
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported apType: $value")
    }
}

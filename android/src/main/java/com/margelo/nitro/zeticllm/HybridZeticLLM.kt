package com.margelo.nitro.zeticllm

import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import com.zeticai.mlange.*
import com.zeticai.mlange.core.cache.ModelCacheHandlingPolicy
import com.zeticai.mlange.core.model.APType
import com.zeticai.mlange.core.model.ModelLoadingStatus
import com.zeticai.mlange.core.model.ModelMode
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
      HybridZeticLLMModel(model, config)
    }
  }

  override fun preloadModel(
    config: NativeLoadModelConfig,
    onProgress: ((NativeModelProgressEvent) -> Unit)?
  ): Promise<Unit> {
    return Promise.async {
      if (config.explicitRuntime != null) {
        throw IllegalArgumentException("INVALID_OPTION: explicitRuntime is not supported for preloadModel.")
      }

      val context = ZeticLLMContextHolder.requireContext()
      val version = config.version?.toInt()
      val cachePolicy = makeCachePolicy(config.cacheHandlingPolicy)

      emitProgress(onProgress, "starting", "auxiliary", config.name, progress = 0.0)

      try {
        val encoder = MLangeDownloadSize.withDownloadedBytes(
          context = context,
          onBytes = makeDownloadedGigabytesCallback(config.name)
        ) {
          ZeticMLangeModel(
            context = context,
            personalKey = config.personalKey,
            name = config.name,
            version = version,
            modelMode = makeGenericModelMode(config.modelMode),
            onProgress = makeAuxiliaryProgressCallback(config.name, onProgress),
            onStatusChanged = makeStatusChangedCallback(config.name, "auxiliary", onProgress),
            cacheHandlingPolicy = cachePolicy
          )
        }
        encoder.close()
        emitProgress(onProgress, "ready", "auxiliary", config.name, progress = 1.0)
      } catch (error: Throwable) {
        emitProgress(
          onProgress,
          "error",
          "auxiliary",
          config.name,
          progress = null,
          error = error.message ?: error.toString()
        )
        throw error
      }
    }
  }

  private fun normalize(value: String?): String = value?.trim()?.uppercase() ?: ""

  private fun makeStatusChangedCallback(
    modelName: String? = null,
    modelRole: String? = null,
    onProgress: ((NativeModelProgressEvent) -> Unit)? = null
  ): (ModelLoadingStatus) -> Unit =
    { status ->
      Log.d(TAG, "model loading status[$modelName/$modelRole]: $status")
      if (normalize(status.name) == "READY" || normalize(status.name) == "LOADED") {
        emitProgress(onProgress, "ready", modelRole, modelName, progress = 1.0)
      }
    }

  private fun makeDownloadedGigabytesCallback(onDownload: ((Double) -> Unit)?): (Long) -> Unit =
    { bytes ->
      val gigabytes = bytes.toDouble() / BYTES_PER_GIGABYTE
      Log.d(TAG, "downloaded model size=${formatGigabytes(gigabytes)} GB")
      onDownload?.invoke(gigabytes)
    }

  private fun makeDownloadedGigabytesCallback(modelName: String): (Long) -> Unit =
    { bytes ->
      val gigabytes = bytes.toDouble() / BYTES_PER_GIGABYTE
      Log.d(TAG, "downloaded auxiliary model size[$modelName]=${formatGigabytes(gigabytes)} GB")
    }

  private fun makeAuxiliaryProgressCallback(
    modelName: String,
    onProgress: ((NativeModelProgressEvent) -> Unit)?
  ): (Float) -> Unit = { progress ->
    emitProgress(
      onProgress,
      "downloading",
      "auxiliary",
      modelName,
      progress = progress.toDouble().coerceIn(0.0, 1.0)
    )
  }

  private fun formatGigabytes(gigabytes: Double): String =
    "%.3f".format(gigabytes)

  private fun makeModelMode(value: String?): LLMModelMode =
    when (normalize(value)) {
      "RUN_SPEED" -> LLMModelMode.RUN_SPEED
      "RUN_ACCURACY" -> LLMModelMode.RUN_ACCURACY
      else -> LLMModelMode.RUN_AUTO
    }

  private fun makeGenericModelMode(value: String?): ModelMode =
    when (normalize(value)) {
      "RUN_SPEED" -> ModelMode.RUN_SPEED
      "RUN_ACCURACY" -> ModelMode.RUN_ACCURACY
      else -> ModelMode.RUN_AUTO
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

  private fun emitProgress(
    callback: ((NativeModelProgressEvent) -> Unit)?,
    phase: String,
    modelRole: String?,
    modelName: String?,
    progress: Double? = null,
    error: String? = null
  ) {
    if (callback == null || modelRole == null || modelName == null) {
      return
    }
    callback(
      NativeModelProgressEvent(
        phase = phase,
        modelRole = modelRole,
        modelName = modelName,
        progress = progress,
        error = error
      )
    )
  }
}

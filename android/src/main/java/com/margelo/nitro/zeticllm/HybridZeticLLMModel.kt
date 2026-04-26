package com.margelo.nitro.zeticllm

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import com.zeticai.mlange.core.cache.ModelCacheHandlingPolicy
import com.zeticai.mlange.core.model.ModelMode
import com.zeticai.mlange.core.model.ZeticMLangeModel
import com.zeticai.mlange.core.model.llm.ZeticMLangeLLMModel
import com.zeticai.mlange.core.model.multimodal.MultimodalProfile
import com.zeticai.mlange.core.model.multimodal.validate
import com.zeticai.mlange.core.tensor.DataType
import com.zeticai.mlange.core.tensor.Tensor
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

@DoNotStrip
class HybridZeticLLMModel(
  private var model: ZeticMLangeLLMModel?,
  private val modelConfig: NativeLoadModelConfig? = null
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

  override fun generateMultimodal(
    config: NativeMultimodalGenerateConfig,
    onToken: ((TokenEvent) -> Unit)?
  ): Promise<GenerateResult> {
    return Promise.async {
      val activeModel = beginGeneration()
      try {
        activeModel.validate(config.profile.toMultimodalProfile())

        val mediaEmbeddings = mutableMapOf<String, FloatArray>()
        config.blocks.forEach { block ->
          val type = block.type.lowercase()
          if (type == "audio" || type == "image") {
            val id = block.id ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: $type block is missing id.")
            if (mediaEmbeddings.containsKey(id)) {
              return@forEach
            }

            val input = block.input
              ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: $type block '$id' is missing input.")
            mediaEmbeddings[id] = when (type) {
              "audio" -> encodeAudioBlock(input, config.audioEncoder, config.audioPreprocess)
              "image" -> encodeImageBlock(input, config.imageEncoder, config.imagePreprocess)
              else -> error("unreachable")
            }
          }
        }

        val embeddings = config.blocks
          .map { block -> block.toEmbeddings(activeModel, mediaEmbeddings) }
          .concatFloatArrays()

        activeModel.runWithEmbeddings(embeddings)
        collectTokens(0, 0, activeModel, onToken)
      } finally {
        endGeneration()
      }
    }
  }

  override fun runWithEmbeddings(
    embeddings: ArrayBuffer,
    onToken: ((TokenEvent) -> Unit)?
  ): Promise<GenerateResult> {
    val ownedEmbeddings = embeddings.asOwning()
    return Promise.async {
      val activeModel = beginGeneration()
      try {
        activeModel.runWithEmbeddings(ownedEmbeddings.toFloatArray())
        collectTokens(0, 0, activeModel, onToken)
      } finally {
        endGeneration()
      }
    }
  }

  override fun tokenize(text: String, parseSpecial: Boolean?): Promise<DoubleArray> {
    return Promise.async {
      val activeModel = currentModel()
      activeModel.tokenize(text, parseSpecial ?: false).map { it.toDouble() }.toDoubleArray()
    }
  }

  override fun tokenEmbeddings(tokenIds: DoubleArray): Promise<ArrayBuffer> {
    return Promise.async {
      val activeModel = currentModel()
      val ids = tokenIds.map { it.toInt() }.toIntArray()
      activeModel.tokenEmbeddings(ids).toArrayBuffer()
    }
  }

  override fun specialTokenId(name: String): Promise<Double> {
    return Promise.async {
      currentModel().specialTokenId(name).toDouble()
    }
  }

  override fun validateMultimodalProfile(profile: NativeMultimodalProfile): Promise<Unit> {
    return Promise.async {
      currentModel().validate(profile.toMultimodalProfile())
    }
  }

  override fun cleanUp(): Promise<Unit> {
    return Promise.async {
      currentModel().cleanUp()
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

  private fun currentModel(): ZeticMLangeLLMModel {
    return synchronized(lock) {
      model ?: throw IllegalStateException("MODEL_RELEASED: This model has already been released.")
    }
  }

  private fun collectTokens(
    promptTokens: Int,
    initialStatus: Int,
    activeModel: ZeticMLangeLLMModel,
    onToken: ((TokenEvent) -> Unit)?
  ): GenerateResult {
    val output = StringBuilder()
    var generatedTokens = 0
    var finalStatus = initialStatus

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

    return GenerateResult(
      text = output.toString(),
      promptTokens = promptTokens.toDouble(),
      generatedTokens = generatedTokens.toDouble(),
      status = finalStatus.toDouble()
    )
  }

  private fun NativePromptEmbeddingBlock.toEmbeddings(
    llm: ZeticMLangeLLMModel,
    mediaEmbeddings: Map<String, FloatArray>
  ): FloatArray {
    return when (type.lowercase()) {
      "text" -> {
        val textValue = text
          ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: text block is missing text.")
        llm.tokenEmbeddings(llm.tokenize(textValue, parseSpecial ?: false))
      }
      "audio", "image" -> {
        val blockId = id
          ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: media block is missing id.")
        mediaEmbeddings[blockId]
          ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: no embeddings were produced for '$blockId'.")
      }
      else -> throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: unsupported block type '$type'.")
    }
  }

  private fun encodeAudioBlock(
    input: NativeMediaInput,
    encoderConfig: NativeMultimodalEncoderConfig?,
    audioPreprocess: String?
  ): FloatArray {
    val config = encoderConfig
      ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: audioEncoder is required for audio blocks.")
    if ((audioPreprocess ?: "qwen-omni-audio") != "qwen-omni-audio") {
      throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: unsupported audioPreprocess '$audioPreprocess'.")
    }
    if (config.inputDataType.lowercase() != "float32") {
      throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: qwen-omni-audio requires a float32 encoder input.")
    }

    val pcm = input.toPcmFloatArray().resampleMonoTo16k()
    val chunks = QwenOmniAudioPreprocessor.melChunks(pcm.samples)
    return runEncoderChunks(config, chunks)
  }

  private fun encodeImageBlock(
    input: NativeMediaInput,
    encoderConfig: NativeMultimodalEncoderConfig?,
    imagePreprocess: NativeImagePreprocessConfig?
  ): FloatArray {
    val config = encoderConfig
      ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: imageEncoder is required for image blocks.")
    val preprocess = imagePreprocess
      ?: throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: imagePreprocess is required for image blocks.")
    val tensor = input.toImageTensor(preprocess, config)
    return runEncoderOnce(config, tensor)
  }

  private fun runEncoderChunks(
    config: NativeMultimodalEncoderConfig,
    chunks: List<TensorPayload>
  ): FloatArray {
    val outputs = mutableListOf<FloatArray>()
    val encoder = createEncoder(config.model)
    try {
      chunks.forEach { payload ->
        outputs += encoder.run(arrayOf(payload.toTensor()))[config.outputIndexValue()].data<FloatArray>()
      }
    } finally {
      encoder.close()
    }
    return outputs.concatFloatArrays()
  }

  private fun runEncoderOnce(
    config: NativeMultimodalEncoderConfig,
    payload: TensorPayload
  ): FloatArray {
    val encoder = createEncoder(config.model)
    try {
      return encoder.run(arrayOf(payload.toTensor()))[config.outputIndexValue()].data<FloatArray>()
    } finally {
      encoder.close()
    }
  }

  private fun createEncoder(config: NativeLoadModelConfig): ZeticMLangeModel {
    if (config.explicitRuntime != null) {
      throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: explicitRuntime is not supported for encoder models yet.")
    }

    val context = ZeticLLMContextHolder.requireContext()
    return ZeticMLangeModel(
      context = context,
      personalKey = config.personalKey,
      name = config.name,
      version = config.version?.toInt(),
      modelMode = when (config.modelMode?.trim()?.uppercase()) {
        "RUN_SPEED" -> ModelMode.RUN_SPEED
        "RUN_ACCURACY" -> ModelMode.RUN_ACCURACY
        else -> ModelMode.RUN_AUTO
      },
      cacheHandlingPolicy = when (config.cacheHandlingPolicy?.trim()?.uppercase()) {
        "KEEP_EXISTING" -> ModelCacheHandlingPolicy.KEEP_EXISTING
        else -> ModelCacheHandlingPolicy.REMOVE_OVERLAPPING
      }
    )
  }

  private fun NativeMultimodalEncoderConfig.outputIndexValue(): Int =
    outputIndex?.toInt() ?: 0

  private fun NativeMultimodalProfile.toMultimodalProfile(): MultimodalProfile =
    MultimodalProfile(
      name = name,
      requiredSpecialTokens = requiredSpecialTokens.toList()
    )

  private fun NativeMediaInput.toPcmFloatArray(): PcmAudio {
    return when (type.lowercase()) {
      "pcm" -> decodePcmBuffer(this)
      "bytes" -> parseWavBytes(data?.toByteArray() ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: bytes input is missing data."))
      "uri" -> parseWavBytes(readUriBytes(uri ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: uri input is missing uri.")))
      else -> throw IllegalArgumentException("INVALID_AUDIO_INPUT: audio blocks accept uri, bytes, or pcm inputs.")
    }
  }

  private fun decodePcmBuffer(input: NativeMediaInput): PcmAudio {
    val sampleRate = input.sampleRate?.toInt()
      ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: pcm input is missing sampleRate.")
    val channels = input.channels?.toInt()
      ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: pcm input is missing channels.")
    val format = input.format?.lowercase()
      ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: pcm input is missing format.")
    val buffer = input.data?.asOwning()?.getBuffer(true)?.order(ByteOrder.LITTLE_ENDIAN)
      ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: pcm input is missing data.")

    val samples = when (format) {
      "float32" -> {
        if (buffer.remaining() % 4 != 0) {
          throw IllegalArgumentException("INVALID_AUDIO_INPUT: float32 pcm byteLength must be divisible by 4.")
        }
        val view = buffer.asFloatBuffer()
        FloatArray(view.remaining()).also { view.get(it) }
      }
      "int16" -> {
        if (buffer.remaining() % 2 != 0) {
          throw IllegalArgumentException("INVALID_AUDIO_INPUT: int16 pcm byteLength must be divisible by 2.")
        }
        val view = buffer.asShortBuffer()
        FloatArray(view.remaining()) { view.get().toFloat() / Short.MAX_VALUE.toFloat() }
      }
      else -> throw IllegalArgumentException("INVALID_AUDIO_INPUT: unsupported pcm format '$format'.")
    }
    return PcmAudio(samples = samples, sampleRate = sampleRate, channels = channels)
  }

  private fun parseWavBytes(bytes: ByteArray): PcmAudio {
    if (bytes.size < 44 || bytes.readAscii(0, 4) != "RIFF" || bytes.readAscii(8, 4) != "WAVE") {
      throw IllegalArgumentException("INVALID_AUDIO_INPUT: uri/bytes audio currently supports PCM WAV data.")
    }

    var offset = 12
    var audioFormat = 0
    var channels = 0
    var sampleRate = 0
    var bitsPerSample = 0
    var dataOffset = -1
    var dataSize = 0

    while (offset + 8 <= bytes.size) {
      val chunkId = bytes.readAscii(offset, 4)
      val chunkSize = bytes.readIntLe(offset + 4)
      val payload = offset + 8
      if (payload + chunkSize > bytes.size) break

      when (chunkId) {
        "fmt " -> {
          audioFormat = bytes.readShortLe(payload)
          channels = bytes.readShortLe(payload + 2)
          sampleRate = bytes.readIntLe(payload + 4)
          bitsPerSample = bytes.readShortLe(payload + 14)
        }
        "data" -> {
          dataOffset = payload
          dataSize = chunkSize
        }
      }

      offset = payload + chunkSize + (chunkSize % 2)
    }

    if (dataOffset < 0 || channels <= 0 || sampleRate <= 0) {
      throw IllegalArgumentException("INVALID_AUDIO_INPUT: WAV data is missing fmt or data chunks.")
    }

    val samples = when {
      audioFormat == 1 && bitsPerSample == 16 -> {
        val count = dataSize / 2
        FloatArray(count) { index ->
          bytes.readShortLe(dataOffset + index * 2).toShort().toFloat() / Short.MAX_VALUE.toFloat()
        }
      }
      audioFormat == 3 && bitsPerSample == 32 -> {
        val count = dataSize / 4
        FloatArray(count) { index ->
          Float.fromBits(bytes.readIntLe(dataOffset + index * 4))
        }
      }
      else -> throw IllegalArgumentException("INVALID_AUDIO_INPUT: WAV must be 16-bit PCM or 32-bit float.")
    }

    return PcmAudio(samples = samples, sampleRate = sampleRate, channels = channels)
  }

  private fun PcmAudio.resampleMonoTo16k(): PcmAudio {
    val mono = if (channels == 1) {
      samples
    } else {
      FloatArray(samples.size / channels) { frame ->
        var sum = 0f
        for (channel in 0 until channels) {
          sum += samples[frame * channels + channel]
        }
        sum / channels
      }
    }

    if (sampleRate == 16_000) {
      return PcmAudio(mono, 16_000, 1)
    }

    val ratio = sampleRate.toDouble() / 16_000.0
    val outputSize = max(1, floor(mono.size / ratio).toInt())
    val resampled = FloatArray(outputSize) { index ->
      val source = index * ratio
      val left = floor(source).toInt().coerceIn(0, mono.lastIndex)
      val right = min(left + 1, mono.lastIndex)
      val fraction = (source - left).toFloat()
      mono[left] * (1f - fraction) + mono[right] * fraction
    }
    return PcmAudio(resampled, 16_000, 1)
  }

  private fun NativeMediaInput.toImageTensor(
    preprocess: NativeImagePreprocessConfig,
    encoder: NativeMultimodalEncoderConfig
  ): TensorPayload {
    val targetWidth = preprocess.width.toInt()
    val targetHeight = preprocess.height.toInt()
    val bitmap = when (type.lowercase()) {
      "pixels" -> pixelsToBitmap(this)
      "bytes" -> {
        val bytes = data?.toByteArray()
          ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: bytes input is missing data.")
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
          ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: could not decode image bytes.")
      }
      "uri" -> decodeBitmapFromUri(uri ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: uri input is missing uri."))
      else -> throw IllegalArgumentException("INVALID_IMAGE_INPUT: image blocks accept uri, bytes, or pixels inputs.")
    }
    val resized = bitmap.resizeTo(targetWidth, targetHeight, preprocess.resizeMode ?: "stretch")
    val shape = encoder.inputShape.map { it.toInt() }.toIntArray()
    val channels = if ((preprocess.colorOrder ?: "rgb").lowercase() == "bgr") {
      intArrayOf(2, 1, 0)
    } else {
      intArrayOf(0, 1, 2)
    }
    val layout = (preprocess.layout ?: "nchw").lowercase()
    val mean = preprocess.mean?.map { it.toFloat() }?.toFloatArray() ?: floatArrayOf(0f, 0f, 0f)
    val std = preprocess.stdValues?.map { it.toFloat() }?.toFloatArray() ?: floatArrayOf(1f, 1f, 1f)
    val dataType = encoder.inputDataType.lowercase()
    val pixelCount = targetWidth * targetHeight

    return when (dataType) {
      "float32" -> {
        val floats = FloatArray(pixelCount * 3)
        for (y in 0 until targetHeight) {
          for (x in 0 until targetWidth) {
            val color = resized.getPixel(x, y)
            val values = floatArrayOf(Color.red(color) / 255f, Color.green(color) / 255f, Color.blue(color) / 255f)
            val pixelIndex = y * targetWidth + x
            for (channel in 0 until 3) {
              val sourceChannel = channels[channel]
              val value = (values[sourceChannel] - mean.getOrElse(channel) { 0f }) / std.getOrElse(channel) { 1f }
              val outIndex = if (layout == "nhwc") pixelIndex * 3 + channel else channel * pixelCount + pixelIndex
              floats[outIndex] = value
            }
          }
        }
        TensorPayload.Float32(floats, shape)
      }
      "uint8", "int8" -> {
        val bytes = ByteArray(pixelCount * 3)
        for (y in 0 until targetHeight) {
          for (x in 0 until targetWidth) {
            val color = resized.getPixel(x, y)
            val values = intArrayOf(Color.red(color), Color.green(color), Color.blue(color))
            val pixelIndex = y * targetWidth + x
            for (channel in 0 until 3) {
              val sourceChannel = channels[channel]
              val value = values[sourceChannel]
              val outIndex = if (layout == "nhwc") pixelIndex * 3 + channel else channel * pixelCount + pixelIndex
              bytes[outIndex] = if (dataType == "int8") (value - 128).toByte() else value.toByte()
            }
          }
        }
        if (dataType == "int8") TensorPayload.Int8(bytes, shape) else TensorPayload.UInt8(bytes, shape)
      }
      else -> throw IllegalArgumentException("INVALID_MULTIMODAL_CONFIG: unsupported encoder inputDataType '$dataType'.")
    }
  }

  private fun pixelsToBitmap(input: NativeMediaInput): Bitmap {
    val width = input.width?.toInt() ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: pixels input is missing width.")
    val height = input.height?.toInt() ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: pixels input is missing height.")
    val format = input.format?.lowercase() ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: pixels input is missing format.")
    val bytes = input.data?.toByteArray() ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: pixels input is missing data.")
    val stride = when (format) {
      "rgba8" -> 4
      "rgb8" -> 3
      else -> throw IllegalArgumentException("INVALID_IMAGE_INPUT: unsupported pixel format '$format'.")
    }
    if (bytes.size < width * height * stride) {
      throw IllegalArgumentException("INVALID_IMAGE_INPUT: pixels buffer is smaller than width * height * channels.")
    }
    val colors = IntArray(width * height) { index ->
      val base = index * stride
      Color.argb(
        if (stride == 4) bytes[base + 3].toInt() and 0xff else 0xff,
        bytes[base].toInt() and 0xff,
        bytes[base + 1].toInt() and 0xff,
        bytes[base + 2].toInt() and 0xff
      )
    }
    return Bitmap.createBitmap(colors, width, height, Bitmap.Config.ARGB_8888)
  }

  private fun decodeBitmapFromUri(uri: String): Bitmap {
    val context = ZeticLLMContextHolder.requireContext()
    val input = if (uri.startsWith("file://") || uri.startsWith("content://")) {
      context.contentResolver.openInputStream(Uri.parse(uri))
    } else {
      context.contentResolver.openInputStream(Uri.fromFile(java.io.File(uri)))
    } ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: could not open image uri '$uri'.")

    input.use {
      return BitmapFactory.decodeStream(it)
        ?: throw IllegalArgumentException("INVALID_IMAGE_INPUT: could not decode image uri '$uri'.")
    }
  }

  private fun Bitmap.resizeTo(width: Int, height: Int, mode: String): Bitmap {
    if (this.width == width && this.height == height && mode == "stretch") {
      return this
    }

    val output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(output)
    canvas.drawColor(Color.BLACK)

    val source = Rect(0, 0, this.width, this.height)
    val target = when (mode.lowercase()) {
      "contain" -> {
        val scale = min(width.toFloat() / this.width, height.toFloat() / this.height)
        val scaledWidth = this.width * scale
        val scaledHeight = this.height * scale
        RectF((width - scaledWidth) / 2f, (height - scaledHeight) / 2f, (width + scaledWidth) / 2f, (height + scaledHeight) / 2f)
      }
      "cover" -> {
        val scale = max(width.toFloat() / this.width, height.toFloat() / this.height)
        val scaledWidth = this.width * scale
        val scaledHeight = this.height * scale
        RectF((width - scaledWidth) / 2f, (height - scaledHeight) / 2f, (width + scaledWidth) / 2f, (height + scaledHeight) / 2f)
      }
      else -> RectF(0f, 0f, width.toFloat(), height.toFloat())
    }
    canvas.drawBitmap(this, source, target, Paint(Paint.FILTER_BITMAP_FLAG))
    return output
  }

  private fun readUriBytes(uri: String): ByteArray {
    val context = ZeticLLMContextHolder.requireContext()
    val input = if (uri.startsWith("file://") || uri.startsWith("content://")) {
      context.contentResolver.openInputStream(Uri.parse(uri))
    } else {
      context.contentResolver.openInputStream(Uri.fromFile(java.io.File(uri)))
    } ?: throw IllegalArgumentException("INVALID_AUDIO_INPUT: could not open uri '$uri'.")

    return input.use {
      val output = ByteArrayOutputStream()
      val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
      while (true) {
        val read = it.read(buffer)
        if (read < 0) break
        output.write(buffer, 0, read)
      }
      output.toByteArray()
    }
  }

  private sealed class TensorPayload(val shape: IntArray, val dataType: DataType) {
    class Float32(val data: FloatArray, shape: IntArray) : TensorPayload(shape, DataType.Float32)
    class UInt8(val data: ByteArray, shape: IntArray) : TensorPayload(shape, DataType.UInt8)
    class Int8(val data: ByteArray, shape: IntArray) : TensorPayload(shape, DataType.Int8)

    fun toTensor(): Tensor = when (this) {
      is Float32 -> Tensor.of(data, dataType, shape)
      is UInt8 -> Tensor.of(data, dataType, shape)
      is Int8 -> Tensor.of(data, dataType, shape)
    }
  }

  private data class PcmAudio(
    val samples: FloatArray,
    val sampleRate: Int,
    val channels: Int
  )

  private object QwenOmniAudioPreprocessor {
    private const val SAMPLE_RATE = 16_000
    private const val CHUNK_SAMPLES = SAMPLE_RATE * 2
    private const val N_FFT = 400
    private const val HOP_LENGTH = 160
    private const val MEL_BINS = 128
    private const val FRAMES_PER_CHUNK = 200
    private val window = FloatArray(N_FFT) { index ->
      (0.5f - 0.5f * cos((2.0 * PI * index) / N_FFT).toFloat())
    }
    private val melFilter = createMelFilter()

    fun melChunks(samples: FloatArray): List<TensorPayload> {
      val totalChunks = max(1, ((samples.size + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES))
      return (0 until totalChunks).map { chunkIndex ->
        val offset = chunkIndex * CHUNK_SAMPLES
        val chunk = FloatArray(CHUNK_SAMPLES) { index ->
          val sampleIndex = offset + index
          if (sampleIndex < samples.size) samples[sampleIndex] else 0f
        }
        TensorPayload.Float32(logMel(chunk), intArrayOf(1, MEL_BINS, FRAMES_PER_CHUNK))
      }
    }

    private fun logMel(samples: FloatArray): FloatArray {
      val output = FloatArray(MEL_BINS * FRAMES_PER_CHUNK)
      val spectrum = FloatArray(N_FFT / 2 + 1)
      val frame = FloatArray(N_FFT)
      for (frameIndex in 0 until FRAMES_PER_CHUNK) {
        val start = frameIndex * HOP_LENGTH - N_FFT / 2
        for (i in 0 until N_FFT) {
          val sampleIndex = start + i
          frame[i] = (if (sampleIndex in samples.indices) samples[sampleIndex] else 0f) * window[i]
        }
        powerSpectrum(frame, spectrum)
        for (mel in 0 until MEL_BINS) {
          var energy = 0f
          for (bin in spectrum.indices) {
            energy += spectrum[bin] * melFilter[mel][bin]
          }
          val logValue = log10(max(energy, 1.0e-10f))
          output[mel * FRAMES_PER_CHUNK + frameIndex] = ((logValue + 4f) / 4f).coerceIn(-1f, 2f)
        }
      }
      return output
    }

    private fun powerSpectrum(frame: FloatArray, output: FloatArray) {
      for (k in output.indices) {
        var real = 0.0
        var imaginary = 0.0
        for (n in frame.indices) {
          val angle = 2.0 * PI * k * n / N_FFT
          real += frame[n] * cos(angle)
          imaginary -= frame[n] * sin(angle)
        }
        output[k] = ((real * real + imaginary * imaginary) / N_FFT).toFloat()
      }
    }

    private fun createMelFilter(): Array<FloatArray> {
      val melMin = hzToMel(0.0)
      val melMax = hzToMel(SAMPLE_RATE / 2.0)
      val points = DoubleArray(MEL_BINS + 2) { index ->
        melToHz(melMin + (melMax - melMin) * index / (MEL_BINS + 1))
      }
      val bins = points.map { hz -> floor((N_FFT + 1) * hz / SAMPLE_RATE).toInt().coerceIn(0, N_FFT / 2) }
      return Array(MEL_BINS) { mel ->
        FloatArray(N_FFT / 2 + 1) { bin ->
          val left = bins[mel]
          val center = bins[mel + 1]
          val right = bins[mel + 2]
          when {
            bin < left || bin > right -> 0f
            bin <= center -> if (center == left) 0f else ((bin - left).toFloat() / (center - left))
            else -> if (right == center) 0f else ((right - bin).toFloat() / (right - center))
          }
        }
      }
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)
    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
  }

  private fun ArrayBuffer.toFloatArray(): FloatArray {
    val byteBuffer = getBuffer(true).order(ByteOrder.nativeOrder())
    if (byteBuffer.remaining() % Float.SIZE_BYTES != 0) {
      throw IllegalArgumentException("INVALID_EMBEDDINGS: embeddings byteLength must be divisible by 4.")
    }
    val view = byteBuffer.asFloatBuffer()
    return FloatArray(view.remaining()).also { view.get(it) }
  }

  private fun FloatArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ByteBuffer.allocateDirect(size * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
    buffer.asFloatBuffer().put(this)
    buffer.rewind()
    return ArrayBuffer.wrap(buffer)
  }

  private fun List<FloatArray>.concatFloatArrays(): FloatArray {
    val totalSize = sumOf { it.size }
    val result = FloatArray(totalSize)
    var offset = 0
    forEach { array ->
      array.copyInto(result, offset)
      offset += array.size
    }
    return result
  }

  private fun ByteArray.readAscii(offset: Int, length: Int): String =
    String(this, offset, length, Charsets.US_ASCII)

  private fun ByteArray.readIntLe(offset: Int): Int =
    (this[offset].toInt() and 0xff) or
      ((this[offset + 1].toInt() and 0xff) shl 8) or
      ((this[offset + 2].toInt() and 0xff) shl 16) or
      ((this[offset + 3].toInt() and 0xff) shl 24)

  private fun ByteArray.readShortLe(offset: Int): Int =
    (this[offset].toInt() and 0xff) or ((this[offset + 1].toInt() and 0xff) shl 8)
}

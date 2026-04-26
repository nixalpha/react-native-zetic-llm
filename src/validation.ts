export type LLMModelMode = 'RUN_AUTO' | 'RUN_SPEED' | 'RUN_ACCURACY'
export type LLMTarget = 'LLAMA_CPP'
export type LLMDataSetType = 'MMLU' | 'TRUTHFULQA' | 'CNN_DAILYMAIL' | 'GSM8K'
export type LLMQuantType =
  | 'GGUF_QUANT_ORG'
  | 'GGUF_QUANT_F16'
  | 'GGUF_QUANT_BF16'
  | 'GGUF_QUANT_Q8_0'
  | 'GGUF_QUANT_Q6_K'
  | 'GGUF_QUANT_Q4_K_M'
  | 'GGUF_QUANT_Q3_K_M'
  | 'GGUF_QUANT_Q2_K'
  | 'GGUF_QUANT_NUM_TYPES'
export type APType = 'CPU' | 'GPU' | 'NPU'
export type LLMKVCacheCleanupPolicy = 'CLEAN_UP_ON_FULL' | 'DO_NOT_CLEAN_UP'
export type CacheHandlingPolicy = 'REMOVE_OVERLAPPING' | 'KEEP_EXISTING'
export type MediaInputType = 'uri' | 'bytes' | 'pcm' | 'pixels'
export type PCMFormat = 'float32' | 'int16'
export type PixelFormat = 'rgba8' | 'rgb8'
export type TensorDataType = 'float32' | 'uint8' | 'int8'
export type AudioPreprocess = 'qwen-omni-audio'
export type ImageResizeMode = 'stretch' | 'contain' | 'cover'
export type ImageColorOrder = 'rgb' | 'bgr'
export type ImageLayout = 'nchw' | 'nhwc'
export type MultimodalMemoryPolicy = 'auto' | 'keepDecoder' | 'swapDecoder'

export interface LLMInitOption {
  kvCacheCleanupPolicy?: LLMKVCacheCleanupPolicy
  nCtx?: number
}

export interface ExplicitRuntimeConfig {
  target: LLMTarget
  quantType: LLMQuantType
  apType?: APType
}

export interface LoadModelConfig {
  personalKey: string
  name: string
  version?: number
  modelMode?: LLMModelMode
  dataSetType?: LLMDataSetType
  cacheHandlingPolicy?: CacheHandlingPolicy
  initOption?: LLMInitOption
  explicitRuntime?: ExplicitRuntimeConfig
}

export type MediaInput =
  | { type: 'uri'; uri: string }
  | { type: 'bytes'; data: ArrayBuffer; mimeType?: string }
  | {
      type: 'pcm'
      data: ArrayBuffer
      sampleRate: number
      channels: number
      format: PCMFormat
    }
  | {
      type: 'pixels'
      data: ArrayBuffer
      width: number
      height: number
      format: PixelFormat
    }

export interface MultimodalEncoderConfig {
  model: LoadModelConfig
  inputShape: number[]
  inputDataType: TensorDataType
  outputIndex?: number
  outputHiddenSize?: number
}

export interface MultimodalProfile {
  name: string
  requiredSpecialTokens: string[]
}

export interface ImagePreprocessConfig {
  width: number
  height: number
  resizeMode?: ImageResizeMode
  colorOrder?: ImageColorOrder
  layout?: ImageLayout
  mean?: number[]
  std?: number[]
}

export type PromptEmbeddingBlock =
  | { type: 'text'; text: string; parseSpecial?: boolean }
  | { type: 'audio'; id: string; input: MediaInput }
  | { type: 'image'; id: string; input: MediaInput }

export interface MultimodalGenerateConfig {
  profile: MultimodalProfile
  audioEncoder?: MultimodalEncoderConfig
  imageEncoder?: MultimodalEncoderConfig
  audioPreprocess?: AudioPreprocess
  imagePreprocess?: ImagePreprocessConfig
  blocks: PromptEmbeddingBlock[]
  memoryPolicy?: MultimodalMemoryPolicy
}

const MODEL_MODES = new Set<LLMModelMode>([
  'RUN_AUTO',
  'RUN_SPEED',
  'RUN_ACCURACY',
])
const DATA_SET_TYPES = new Set<LLMDataSetType>([
  'MMLU',
  'TRUTHFULQA',
  'CNN_DAILYMAIL',
  'GSM8K',
])
const TARGETS = new Set<LLMTarget>(['LLAMA_CPP'])
const QUANT_TYPES = new Set<LLMQuantType>([
  'GGUF_QUANT_ORG',
  'GGUF_QUANT_F16',
  'GGUF_QUANT_BF16',
  'GGUF_QUANT_Q8_0',
  'GGUF_QUANT_Q6_K',
  'GGUF_QUANT_Q4_K_M',
  'GGUF_QUANT_Q3_K_M',
  'GGUF_QUANT_Q2_K',
  'GGUF_QUANT_NUM_TYPES',
])
const AP_TYPES = new Set<APType>(['CPU', 'GPU', 'NPU'])
const KV_POLICIES = new Set<LLMKVCacheCleanupPolicy>([
  'CLEAN_UP_ON_FULL',
  'DO_NOT_CLEAN_UP',
])
const CACHE_POLICIES = new Set<CacheHandlingPolicy>([
  'REMOVE_OVERLAPPING',
  'KEEP_EXISTING',
])
const MEDIA_INPUT_TYPES = new Set<MediaInputType>(['uri', 'bytes', 'pcm', 'pixels'])
const PCM_FORMATS = new Set<PCMFormat>(['float32', 'int16'])
const PIXEL_FORMATS = new Set<PixelFormat>(['rgba8', 'rgb8'])
const TENSOR_DATA_TYPES = new Set<TensorDataType>(['float32', 'uint8', 'int8'])
const AUDIO_PREPROCESSORS = new Set<AudioPreprocess>(['qwen-omni-audio'])
const IMAGE_RESIZE_MODES = new Set<ImageResizeMode>(['stretch', 'contain', 'cover'])
const IMAGE_COLOR_ORDERS = new Set<ImageColorOrder>(['rgb', 'bgr'])
const IMAGE_LAYOUTS = new Set<ImageLayout>(['nchw', 'nhwc'])
const MEMORY_POLICIES = new Set<MultimodalMemoryPolicy>([
  'auto',
  'keepDecoder',
  'swapDecoder',
])

function assertString(value: string | undefined, field: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`react-native-zetic-llm: ${field} must be a non-empty string.`)
  }
}

function assertArrayBuffer(value: unknown, field: string): asserts value is ArrayBuffer {
  if (
    value == null ||
    typeof value !== 'object' ||
    typeof (value as ArrayBuffer).byteLength !== 'number'
  ) {
    throw new Error(`react-native-zetic-llm: ${field} must be an ArrayBuffer.`)
  }
}

function assertEnum<T extends string>(
  value: T | undefined,
  allowed: Set<T>,
  field: string
) {
  if (value != null && !allowed.has(value)) {
    throw new Error(
      `react-native-zetic-llm: ${field} must be one of ${Array.from(allowed).join(', ')}.`
    )
  }
}

function assertPositiveInteger(value: number | undefined, field: string) {
  if (value == null || !Number.isInteger(value) || value <= 0) {
    throw new Error(`react-native-zetic-llm: ${field} must be a positive integer.`)
  }
}

function assertNumberArray(value: number[] | undefined, field: string, required = false) {
  if (value == null) {
    if (required) {
      throw new Error(`react-native-zetic-llm: ${field} is required.`)
    }
    return
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'number' || !Number.isFinite(item))) {
    throw new Error(`react-native-zetic-llm: ${field} must be an array of finite numbers.`)
  }
}

export function validateLoadModelConfig(config: LoadModelConfig): LoadModelConfig {
  if (config == null || typeof config !== 'object') {
    throw new Error('react-native-zetic-llm: config is required.')
  }

  assertString(config.personalKey, 'personalKey')
  assertString(config.name, 'name')

  if (config.version != null && (!Number.isInteger(config.version) || config.version <= 0)) {
    throw new Error('react-native-zetic-llm: version must be a positive integer.')
  }

  assertEnum(config.modelMode, MODEL_MODES, 'modelMode')
  assertEnum(config.dataSetType, DATA_SET_TYPES, 'dataSetType')
  assertEnum(config.cacheHandlingPolicy, CACHE_POLICIES, 'cacheHandlingPolicy')
  assertEnum(config.initOption?.kvCacheCleanupPolicy, KV_POLICIES, 'initOption.kvCacheCleanupPolicy')

  if (config.initOption?.nCtx != null) {
    if (!Number.isInteger(config.initOption.nCtx) || config.initOption.nCtx <= 0) {
      throw new Error('react-native-zetic-llm: initOption.nCtx must be a positive integer.')
    }
  }

  if (config.explicitRuntime != null) {
    assertEnum(config.explicitRuntime.target, TARGETS, 'explicitRuntime.target')
    assertEnum(config.explicitRuntime.quantType, QUANT_TYPES, 'explicitRuntime.quantType')
    assertEnum(config.explicitRuntime.apType, AP_TYPES, 'explicitRuntime.apType')
  }

  return config
}

export function validateMultimodalProfile(profile: MultimodalProfile): MultimodalProfile {
  if (profile == null || typeof profile !== 'object') {
    throw new Error('react-native-zetic-llm: profile is required.')
  }

  assertString(profile.name, 'profile.name')
  if (!Array.isArray(profile.requiredSpecialTokens)) {
    throw new Error('react-native-zetic-llm: profile.requiredSpecialTokens must be an array.')
  }
  profile.requiredSpecialTokens.forEach((token, index) => {
    assertString(token, `profile.requiredSpecialTokens[${index}]`)
  })

  return profile
}

export function validateMultimodalEncoderConfig(
  config: MultimodalEncoderConfig,
  field: string
): MultimodalEncoderConfig {
  if (config == null || typeof config !== 'object') {
    throw new Error(`react-native-zetic-llm: ${field} must be an object.`)
  }

  validateLoadModelConfig(config.model)
  assertNumberArray(config.inputShape, `${field}.inputShape`, true)
  config.inputShape.forEach((dimension, index) => {
    assertPositiveInteger(dimension, `${field}.inputShape[${index}]`)
  })
  assertString(config.inputDataType, `${field}.inputDataType`)
  assertEnum(config.inputDataType, TENSOR_DATA_TYPES, `${field}.inputDataType`)
  if (config.outputIndex != null) {
    if (!Number.isInteger(config.outputIndex) || config.outputIndex < 0) {
      throw new Error(`react-native-zetic-llm: ${field}.outputIndex must be a non-negative integer.`)
    }
  }
  if (config.outputHiddenSize != null) {
    assertPositiveInteger(config.outputHiddenSize, `${field}.outputHiddenSize`)
  }

  return config
}

export function validateMediaInput(input: MediaInput, field: string): MediaInput {
  if (input == null || typeof input !== 'object') {
    throw new Error(`react-native-zetic-llm: ${field} must be an object.`)
  }

  assertEnum(input.type, MEDIA_INPUT_TYPES, `${field}.type`)
  switch (input.type) {
    case 'uri':
      assertString(input.uri, `${field}.uri`)
      break
    case 'bytes':
      assertArrayBuffer(input.data, `${field}.data`)
      if (input.mimeType != null) {
        assertString(input.mimeType, `${field}.mimeType`)
      }
      break
    case 'pcm':
      assertArrayBuffer(input.data, `${field}.data`)
      assertPositiveInteger(input.sampleRate, `${field}.sampleRate`)
      assertPositiveInteger(input.channels, `${field}.channels`)
      assertEnum(input.format, PCM_FORMATS, `${field}.format`)
      break
    case 'pixels':
      assertArrayBuffer(input.data, `${field}.data`)
      assertPositiveInteger(input.width, `${field}.width`)
      assertPositiveInteger(input.height, `${field}.height`)
      assertEnum(input.format, PIXEL_FORMATS, `${field}.format`)
      break
  }

  return input
}

export function validateImagePreprocessConfig(
  config: ImagePreprocessConfig | undefined
): ImagePreprocessConfig | undefined {
  if (config == null) {
    return undefined
  }
  if (typeof config !== 'object') {
    throw new Error('react-native-zetic-llm: imagePreprocess must be an object.')
  }

  assertPositiveInteger(config.width, 'imagePreprocess.width')
  assertPositiveInteger(config.height, 'imagePreprocess.height')
  assertEnum(config.resizeMode, IMAGE_RESIZE_MODES, 'imagePreprocess.resizeMode')
  assertEnum(config.colorOrder, IMAGE_COLOR_ORDERS, 'imagePreprocess.colorOrder')
  assertEnum(config.layout, IMAGE_LAYOUTS, 'imagePreprocess.layout')
  assertNumberArray(config.mean, 'imagePreprocess.mean')
  assertNumberArray(config.std, 'imagePreprocess.std')

  return config
}

export function validateMultimodalGenerateConfig(
  config: MultimodalGenerateConfig
): MultimodalGenerateConfig {
  if (config == null || typeof config !== 'object') {
    throw new Error('react-native-zetic-llm: multimodal config is required.')
  }

  validateMultimodalProfile(config.profile)
  if (config.audioEncoder != null) {
    validateMultimodalEncoderConfig(config.audioEncoder, 'audioEncoder')
  }
  if (config.imageEncoder != null) {
    validateMultimodalEncoderConfig(config.imageEncoder, 'imageEncoder')
  }
  assertEnum(config.audioPreprocess, AUDIO_PREPROCESSORS, 'audioPreprocess')
  validateImagePreprocessConfig(config.imagePreprocess)
  assertEnum(config.memoryPolicy, MEMORY_POLICIES, 'memoryPolicy')

  if (!Array.isArray(config.blocks) || config.blocks.length === 0) {
    throw new Error('react-native-zetic-llm: blocks must be a non-empty array.')
  }

  config.blocks.forEach((block, index) => {
    if (block == null || typeof block !== 'object') {
      throw new Error(`react-native-zetic-llm: blocks[${index}] must be an object.`)
    }

    switch (block.type) {
      case 'text':
        assertString(block.text, `blocks[${index}].text`)
        break
      case 'audio':
        assertString(block.id, `blocks[${index}].id`)
        validateMediaInput(block.input, `blocks[${index}].input`)
        if (config.audioEncoder == null) {
          throw new Error('react-native-zetic-llm: audioEncoder is required when blocks include audio.')
        }
        break
      case 'image':
        assertString(block.id, `blocks[${index}].id`)
        validateMediaInput(block.input, `blocks[${index}].input`)
        if (config.imageEncoder == null) {
          throw new Error('react-native-zetic-llm: imageEncoder is required when blocks include image.')
        }
        if (config.imagePreprocess == null) {
          throw new Error('react-native-zetic-llm: imagePreprocess is required when blocks include image.')
        }
        break
      default:
        throw new Error(`react-native-zetic-llm: blocks[${index}].type is unsupported.`)
    }
  })

  return config
}

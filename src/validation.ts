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

function assertString(value: string | undefined, field: string): asserts value is string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`react-native-zetic-llm: ${field} must be a non-empty string.`)
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

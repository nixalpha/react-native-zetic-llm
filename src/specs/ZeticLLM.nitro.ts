import type { HybridObject } from 'react-native-nitro-modules'

export interface NativeLLMInitOption {
  kvCacheCleanupPolicy?: string
  nCtx?: number
}

export interface NativeExplicitRuntimeConfig {
  target: string
  quantType: string
  apType?: string
}

export interface NativeLoadModelConfig {
  personalKey: string
  name: string
  version?: number
  modelMode?: string
  dataSetType?: string
  cacheHandlingPolicy?: string
  initOption?: NativeLLMInitOption
  explicitRuntime?: NativeExplicitRuntimeConfig
}

export interface NativeMediaInput {
  type: string
  uri?: string
  data?: ArrayBuffer
  mimeType?: string
  sampleRate?: number
  channels?: number
  format?: string
  width?: number
  height?: number
}

export interface NativeMultimodalEncoderConfig {
  model: NativeLoadModelConfig
  inputShape: number[]
  inputDataType: string
  outputIndex?: number
  outputHiddenSize?: number
}

export interface NativeMultimodalProfile {
  name: string
  requiredSpecialTokens: string[]
}

export interface NativeImagePreprocessConfig {
  width: number
  height: number
  resizeMode?: string
  colorOrder?: string
  layout?: string
  mean?: number[]
  std?: number[]
}

export interface NativePromptEmbeddingBlock {
  type: string
  text?: string
  parseSpecial?: boolean
  id?: string
  input?: NativeMediaInput
}

export interface NativeMultimodalGenerateConfig {
  profile: NativeMultimodalProfile
  audioEncoder?: NativeMultimodalEncoderConfig
  imageEncoder?: NativeMultimodalEncoderConfig
  audioPreprocess?: string
  imagePreprocess?: NativeImagePreprocessConfig
  blocks: NativePromptEmbeddingBlock[]
  memoryPolicy?: string
}

export interface TokenEvent {
  token: string
  generatedTokens: number
  status: number
}

export interface GenerateResult {
  text: string
  promptTokens: number
  generatedTokens: number
  status: number
}

export interface ZeticLLMModel
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  generate(prompt: string, onToken?: (event: TokenEvent) => void): Promise<GenerateResult>
  generateMultimodal(
    config: NativeMultimodalGenerateConfig,
    onToken?: (event: TokenEvent) => void
  ): Promise<GenerateResult>
  runWithEmbeddings(
    embeddings: ArrayBuffer,
    onToken?: (event: TokenEvent) => void
  ): Promise<GenerateResult>
  tokenize(text: string, parseSpecial?: boolean): Promise<number[]>
  tokenEmbeddings(tokenIds: number[]): Promise<ArrayBuffer>
  specialTokenId(name: string): Promise<number>
  validateMultimodalProfile(profile: NativeMultimodalProfile): Promise<void>
  cleanUp(): Promise<void>
  release(): void
}

export interface ZeticLLM
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  loadModel(
    config: NativeLoadModelConfig,
    onDownload?: (progress: number) => void
  ): Promise<ZeticLLMModel>
}

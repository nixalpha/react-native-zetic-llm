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

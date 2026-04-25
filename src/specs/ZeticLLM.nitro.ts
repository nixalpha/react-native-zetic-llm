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

export interface AgentOptions {
  maxActions?: number
  maxRuntimeMs?: number
  requireConfirmation?: boolean
  useShadowTreeSnapshot?: boolean
}

export interface AgentEvent {
  type: string
  state: string
  message: string
  snapshot?: string
  actionJson?: string
  progress?: number
}

export interface AgentStateSnapshot {
  state: string
  task: string
  instruction: string
  actionsTaken: number
  memory: string
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

export interface ZeticAgent
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  loadModel(
    config: NativeLoadModelConfig,
    onDownload?: (progress: number) => void
  ): Promise<void>
  start(task: string, options?: AgentOptions): Promise<void>
  pause(): void
  resume(): void
  stop(): void
  sendInstruction(text: string): void
  snapshot(): Promise<string>
  getState(): AgentStateSnapshot
  onEvent(callback: (event: AgentEvent) => void): void
}

export interface ZeticAgentShadowTree
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  getLatestSnapshot(): string
  setFallbackSnapshot(snapshot: string): void
  clear(): void
}

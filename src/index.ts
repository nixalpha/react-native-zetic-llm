import { NitroModules } from 'react-native-nitro-modules'

export type {
  AgentEvent,
  AgentOptions,
  AgentStateSnapshot,
  GenerateResult,
  TokenEvent,
  ZeticAgent as NativeZeticAgent,
  ZeticAgentShadowTree,
  ZeticLLMModel,
} from './specs/ZeticLLM.nitro'

import type {
  AgentOptions,
  NativeLoadModelConfig,
  ZeticAgent as NativeZeticAgent,
  ZeticAgentShadowTree,
  ZeticLLM as NativeZeticLLM,
  ZeticLLMModel,
} from './specs/ZeticLLM.nitro'
export {
  validateLoadModelConfig,
  type APType,
  type CacheHandlingPolicy,
  type ExplicitRuntimeConfig,
  type LLMDataSetType,
  type LLMInitOption,
  type LLMKVCacheCleanupPolicy,
  type LLMModelMode,
  type LLMQuantType,
  type LLMTarget,
  type LoadModelConfig,
} from './validation'
import { validateLoadModelConfig, type LoadModelConfig } from './validation'

export interface ZeticLLM {
  loadModel(
    config: LoadModelConfig,
    onDownload?: (progress: number) => void
  ): Promise<ZeticLLMModel>
}

export interface ZeticAgent {
  loadModel(
    config: LoadModelConfig,
    onDownload?: (progress: number) => void
  ): Promise<void>
  start(task: string, options?: AgentOptions): Promise<void>
  pause(): void
  resume(): void
  stop(): void
  sendInstruction(text: string): void
  snapshot(): Promise<string>
  getState(): ReturnType<NativeZeticAgent['getState']>
  onEvent(callback: Parameters<NativeZeticAgent['onEvent']>[0]): void
}

export type ZeticAgentShadowTreeController = ZeticAgentShadowTree & {
  installShadowTreeObserver?: () => boolean
}

let cachedNativeModule: NativeZeticLLM | undefined
let cachedModule: ZeticLLM | undefined
let cachedNativeAgent: NativeZeticAgent | undefined
let cachedAgent: ZeticAgent | undefined
let cachedShadowTree: ZeticAgentShadowTreeController | undefined

export function createZeticLLM(): ZeticLLM {
  if (cachedModule == null) {
    cachedNativeModule ??= NitroModules.createHybridObject<NativeZeticLLM>('ZeticLLM')
    cachedModule = {
      loadModel(config, onDownload) {
        return cachedNativeModule!.loadModel(
          validateLoadModelConfig(config) as NativeLoadModelConfig,
          onDownload
        )
      },
    }
  }
  return cachedModule
}

export function loadModel(
  config: LoadModelConfig,
  onDownload?: (progress: number) => void
) {
  return createZeticLLM().loadModel(config, onDownload)
}

export function createZeticAgent(): ZeticAgent {
  if (cachedAgent == null) {
    cachedNativeAgent ??= NitroModules.createHybridObject<NativeZeticAgent>('ZeticAgent')
    cachedAgent = {
      loadModel(config, onDownload) {
        return cachedNativeAgent!.loadModel(
          validateLoadModelConfig(config) as NativeLoadModelConfig,
          onDownload
        )
      },
      start(task, options) {
        return cachedNativeAgent!.start(task, options)
      },
      pause() {
        cachedNativeAgent!.pause()
      },
      resume() {
        cachedNativeAgent!.resume()
      },
      stop() {
        cachedNativeAgent!.stop()
      },
      sendInstruction(text) {
        cachedNativeAgent!.sendInstruction(text)
      },
      snapshot() {
        return cachedNativeAgent!.snapshot()
      },
      getState() {
        return cachedNativeAgent!.getState()
      },
      onEvent(callback) {
        cachedNativeAgent!.onEvent(callback)
      },
    }
  }
  return cachedAgent
}

export function createZeticAgentShadowTree(): ZeticAgentShadowTreeController {
  cachedShadowTree ??= NitroModules.createHybridObject<ZeticAgentShadowTree>(
    'ZeticAgentShadowTree'
  ) as ZeticAgentShadowTreeController
  return cachedShadowTree
}

export function installZeticAgentShadowTreeObserver(): boolean {
  return createZeticAgentShadowTree().installShadowTreeObserver?.() ?? false
}

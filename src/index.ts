import { NitroModules } from 'react-native-nitro-modules'

export type { GenerateResult, TokenEvent, ZeticLLMModel } from './specs/ZeticLLM.nitro'

import type {
  NativeLoadModelConfig,
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

let cachedNativeModule: NativeZeticLLM | undefined
let cachedModule: ZeticLLM | undefined

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

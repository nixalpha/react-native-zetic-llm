import Foundation
import NitroModules
import ZeticMLange

final class HybridZeticLLM: HybridZeticLLMSpec {
  func loadModel(
    config: NativeLoadModelConfig,
    onDownload: ((Double) -> Void)?
  ) throws -> Promise<(any HybridZeticLLMModelSpec)> {
    return Promise.async {
      let initOption = Self.makeInitOption(config.initOption)
      let version = config.version.map { Int($0) }
      let cachePolicy = Self.makeCachePolicy(config.cacheHandlingPolicy)
      let progress: ((Float) -> Void)? = onDownload.map { callback in
        { value in callback(Double(value)) }
      }

      let nativeModel: ZeticMLangeLLMModel
      if let explicitRuntime = config.explicitRuntime {
        nativeModel = try ZeticMLangeLLMModel(
          personalKey: config.personalKey,
          name: config.name,
          version: version,
          target: try Self.makeTarget(explicitRuntime.target),
          quantType: try Self.makeQuantType(explicitRuntime.quantType),
          apType: try Self.makeAPType(explicitRuntime.apType),
          cacheHandlingPolicy: cachePolicy,
          initOption: initOption,
          onDownload: progress
        )
      } else {
        nativeModel = try ZeticMLangeLLMModel(
          personalKey: config.personalKey,
          name: config.name,
          version: version,
          modelMode: Self.makeModelMode(config.modelMode),
          dataSetType: try Self.makeDataSetType(config.dataSetType),
          cacheHandlingPolicy: cachePolicy,
          initOption: initOption,
          onDownload: progress
        )
      }

      return HybridZeticLLMModel(model: nativeModel)
    }
  }

  private static func normalize(_ value: String?) -> String {
    return (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
  }

  private static func makeModelMode(_ value: String?) -> LLMModelMode {
    switch normalize(value) {
    case "RUN_SPEED":
      return .RUN_SPEED
    case "RUN_ACCURACY":
      return .RUN_ACCURACY
    default:
      return .RUN_AUTO
    }
  }

  private static func makeDataSetType(_ value: String?) throws -> LLMDataSetType? {
    switch normalize(value) {
    case "":
      return nil
    case "MMLU":
      return .MMLU
    case "TRUTHFULQA":
      return .TRUTHFULQA
    case "CNN_DAILYMAIL":
      return .CNN_DAILYMAIL
    case "GSM8K":
      return .GSM8K
    default:
      throw ZeticLLMError.invalidOption("Unsupported dataSetType: \(value ?? "")")
    }
  }

  private static func makeCachePolicy(_ value: String?) -> ZeticMLangeCacheHandlingPolicy {
    switch normalize(value) {
    case "KEEP_EXISTING":
      return .KEEP_EXISTING
    default:
      return .REMOVE_OVERLAPPING
    }
  }

  private static func makeKVPolicy(_ value: String?) -> LLMKVCacheCleanupPolicy {
    switch normalize(value) {
    case "DO_NOT_CLEAN_UP":
      return .DO_NOT_CLEAN_UP
    default:
      return .CLEAN_UP_ON_FULL
    }
  }

  private static func makeInitOption(_ option: NativeLLMInitOption?) -> LLMInitOption {
    return LLMInitOption(
      kvCacheCleanupPolicy: makeKVPolicy(option?.kvCacheCleanupPolicy),
      nCtx: option?.nCtx.map { Int($0) } ?? 2048
    )
  }

  private static func makeTarget(_ value: String) throws -> LLMTarget {
    switch normalize(value) {
    case "LLAMA_CPP":
      return .LLAMA_CPP
    default:
      throw ZeticLLMError.invalidOption("Unsupported target: \(value)")
    }
  }

  private static func makeQuantType(_ value: String) throws -> LLMQuantType {
    switch normalize(value) {
    case "GGUF_QUANT_ORG":
      return .GGUF_QUANT_ORG
    case "GGUF_QUANT_F16":
      return .GGUF_QUANT_F16
    case "GGUF_QUANT_BF16":
      return .GGUF_QUANT_BF16
    case "GGUF_QUANT_Q8_0":
      return .GGUF_QUANT_Q8_0
    case "GGUF_QUANT_Q6_K":
      return .GGUF_QUANT_Q6_K
    case "GGUF_QUANT_Q4_K_M":
      return .GGUF_QUANT_Q4_K_M
    case "GGUF_QUANT_Q3_K_M":
      return .GGUF_QUANT_Q3_K_M
    case "GGUF_QUANT_Q2_K":
      return .GGUF_QUANT_Q2_K
    case "GGUF_QUANT_NUM_TYPES":
      return .GGUF_QUANT_NUM_TYPES
    default:
      throw ZeticLLMError.invalidOption("Unsupported quantType: \(value)")
    }
  }

  private static func makeAPType(_ value: String?) throws -> APType {
    switch normalize(value) {
    case "", "CPU":
      return .CPU
    case "GPU":
      return .GPU
    case "NPU":
      throw ZeticLLMError.invalidOption("APType.NPU is not supported by ZeticMLangeLLMModel on iOS.")
    default:
      throw ZeticLLMError.invalidOption("Unsupported apType: \(value ?? "")")
    }
  }
}

enum ZeticLLMError: Error, LocalizedError {
  case invalidOption(String)
  case modelReleased
  case generationInProgress

  var errorDescription: String? {
    switch self {
    case .invalidOption(let message):
      return "INVALID_OPTION: \(message)"
    case .modelReleased:
      return "MODEL_RELEASED: This model has already been released."
    case .generationInProgress:
      return "GENERATION_IN_PROGRESS: This model is already generating."
    }
  }
}

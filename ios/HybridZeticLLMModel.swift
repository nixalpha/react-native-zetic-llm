import Foundation
import NitroModules
import ZeticMLange

final class HybridZeticLLMModel: HybridZeticLLMModelSpec {
  private let lock = NSLock()
  private var model: ZeticMLangeLLMModel?
  private var isGenerating = false

  init(model: ZeticMLangeLLMModel) {
    self.model = model
  }

  func generate(
    prompt: String,
    onToken: ((TokenEvent) -> Void)?
  ) throws -> Promise<GenerateResult> {
    return Promise.async {
      let activeModel = try self.beginGeneration()
      defer { self.endGeneration() }

      let runResult = try activeModel.run(prompt)
      var output = ""
      var generatedTokens = 0
      var finalStatus = 0

      while true {
        let next = activeModel.waitForNextToken()
        finalStatus = next.code
        let count = next.generatedTokens
        if count == 0 {
          break
        }

        generatedTokens = count
        output.append(next.token)
        onToken?(
          TokenEvent(
            token: next.token,
            generatedTokens: Double(count),
            status: Double(next.code)
          )
        )
      }

      return GenerateResult(
        text: output,
        promptTokens: Double(runResult.promptTokens),
        generatedTokens: Double(generatedTokens),
        status: Double(finalStatus)
      )
    }
  }

  func cleanUp() throws -> Promise<Void> {
    return Promise.async {
      guard let model = self.currentModel() else {
        throw ZeticLLMError.modelReleased
      }
      try model.cleanUp()
    }
  }

  func release() throws {
    lock.lock()
    let releasedModel = model
    model = nil
    isGenerating = false
    lock.unlock()

    releasedModel?.forceDeinit()
  }

  private func beginGeneration() throws -> ZeticMLangeLLMModel {
    lock.lock()
    defer { lock.unlock() }

    guard let model else {
      throw ZeticLLMError.modelReleased
    }
    if isGenerating {
      throw ZeticLLMError.generationInProgress
    }
    isGenerating = true
    return model
  }

  private func endGeneration() {
    lock.lock()
    isGenerating = false
    lock.unlock()
  }

  private func currentModel() -> ZeticMLangeLLMModel? {
    lock.lock()
    defer { lock.unlock() }
    return model
  }
}

import Foundation
import NitroModules
import UIKit
import ZeticMLange

final class HybridZeticAgent: HybridZeticAgentSpec {
  private let stateQueue = DispatchQueue(label: "com.margelo.nitro.zeticllm.agent.state")
  private var model: ZeticMLangeLLMModel?
  private var listener: ((AgentEvent) -> Void)?
  private var state = "idle"
  private var task = ""
  private var instruction = ""
  private var memory = ""
  private var actionsTaken = 0
  private var stopRequested = false
  private var paused = false
  private var isRunning = false
  private var nodeRects: [String: CGRect] = [:]

  func loadModel(
    config: NativeLoadModelConfig,
    onDownload: ((Double) -> Void)?
  ) throws -> Promise<Void> {
    return Promise.async {
      self.setState("loading_model")
      let progress: ((Float) -> Void)? = { value in
        let progress = Double(value).clamped(to: 0...1)
        self.emit(type: "model_progress", message: String(format: "Model download %.1f%%", progress * 100), progress: progress)
        onDownload?(progress)
      }

      let initOption = Self.makeInitOption(config.initOption)
      let version = config.version.map { Int($0) }
      let cachePolicy = Self.makeCachePolicy(config.cacheHandlingPolicy)
      let loaded: ZeticMLangeLLMModel

      if let explicitRuntime = config.explicitRuntime {
        loaded = try ZeticMLangeLLMModel(
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
        loaded = try ZeticMLangeLLMModel(
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

      let oldModel = self.stateQueue.sync {
        let previous = self.model
        self.model = loaded
        return previous
      }
      oldModel?.forceDeinit()

      self.setState("idle")
      self.emit(type: "model_loaded", message: "Zetic agent model loaded.")
    }
  }

  func start(task: String, options: AgentOptions?) throws -> Promise<Void> {
    return Promise.async {
      let activeModel: ZeticMLangeLLMModel = try self.stateQueue.sync {
        guard let activeModel = self.model else {
          throw ZeticLLMError.invalidOption("MODEL_NOT_LOADED: Call loadModel() before start().")
        }
        if self.isRunning {
          throw ZeticLLMError.invalidOption("AGENT_RUNNING: The agent is already running.")
        }
        self.task = task
        self.actionsTaken = 0
        self.stopRequested = false
        self.paused = false
        self.isRunning = true
        return activeModel
      }

      let maxActions = max(1, Int(options?.maxActions ?? 12))
      let maxRuntimeMs = max(1_000, Int(options?.maxRuntimeMs ?? 120_000))
      let deadline = Date().addingTimeInterval(Double(maxRuntimeMs) / 1_000)

      defer {
        self.stateQueue.sync {
          self.isRunning = false
        }
        self.setState(self.stopRequested ? "stopped" : "idle")
        self.emit(type: "stopped", message: "Agent stopped after \(self.actionsTaken) action(s).")
      }

      self.emit(type: "started", message: "Agent started: \(task)")

      while !self.stopRequested && self.actionsTaken < maxActions && Date() < deadline {
        self.waitWhilePaused()
        if self.stopRequested { break }

        self.setState("observing")
        let screen = try self.snapshotBlocking()
        self.emit(type: "snapshot", message: "Captured UI snapshot.", snapshot: screen)

        self.setState("thinking")
        let response = try self.generatePlan(activeModel, prompt: self.buildPrompt(task: task, snapshot: screen))
        self.emit(type: "llm_output", message: "LLM produced an action plan.", actionJson: response)

        let action = self.parseAction(response)
        if options?.requireConfirmation == true && action["type"] as? String != "done" {
          self.setState("paused")
          self.paused = true
          self.emit(type: "confirmation_required", message: "Action requires user confirmation.", actionJson: self.jsonString(action))
          self.waitWhilePaused()
        }

        if self.stopRequested { break }
        self.setState("acting")
        let outcome = try self.executeAction(action)
        if outcome.consumedAction {
          self.actionsTaken += 1
        }
        self.memory = String((self.memory + "\n- \(action["type"] as? String ?? "unknown"): \(outcome.message)").suffix(4_000))
        self.emit(type: "action", message: outcome.message, actionJson: self.jsonString(action))

        if outcome.done { break }
        Self.sleep(milliseconds: 350)
      }
    }
  }

  func pause() throws {
    paused = true
    setState("paused")
    emit(type: "paused", message: "Agent paused.")
  }

  func resume() throws {
    paused = false
    setState(isRunning ? "observing" : "idle")
    emit(type: "resumed", message: "Agent resumed.")
  }

  func stop() throws {
    stopRequested = true
    paused = false
    setState("stopped")
    emit(type: "stop_requested", message: "Stop requested.")
  }

  func sendInstruction(text: String) throws {
    instruction = text
    emit(type: "instruction", message: "User instruction updated: \(text)")
  }

  func snapshot() throws -> Promise<String> {
    return Promise.async {
      try self.snapshotBlocking()
    }
  }

  func getState() throws -> AgentStateSnapshot {
    return AgentStateSnapshot(
      state: state,
      task: task,
      instruction: instruction,
      actionsTaken: Double(actionsTaken),
      memory: memory
    )
  }

  func onEvent(callback: @escaping (AgentEvent) -> Void) throws {
    listener = callback
  }

  private struct ActionOutcome {
    let message: String
    let done: Bool
    let consumedAction: Bool
  }

  private func generatePlan(_ model: ZeticMLangeLLMModel, prompt: String) throws -> String {
    try model.cleanUp()
    _ = try model.run(prompt)
    var output = ""
    while !stopRequested {
      let next = model.waitForNextToken()
      if next.generatedTokens == 0 {
        break
      }
      output.append(next.token)
    }
    try model.cleanUp()
    return output
  }

  private func buildPrompt(task: String, snapshot: String) -> String {
    return """
    You are an on-device React Native UI agent. Decide exactly one next action.
    Task: \(task)
    Latest user instruction: \(instruction.isEmpty ? "(none)" : instruction)
    Recent memory:
    \(memory.isEmpty ? "(none)" : memory)

    UI snapshot:
    \(snapshot)

    Return strict JSON only, no Markdown:
    {"type":"tap","nodeId":"node id from snapshot"}
    {"type":"tap","x":123,"y":456}
    {"type":"wait","ms":500}
    {"type":"askUser","question":"short question"}
    {"type":"done","reason":"why the task is complete"}
    """
  }

  private func parseAction(_ text: String) -> [String: Any] {
    guard
      let start = text.firstIndex(of: "{"),
      let end = text.lastIndex(of: "}"),
      start < end
    else {
      return ["type": "askUser", "question": "The model did not return a JSON action."]
    }
    let jsonText = String(text[start...end])
    guard
      let data = jsonText.data(using: .utf8),
      let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      return ["type": "askUser", "question": "The model returned invalid JSON."]
    }
    return parsed["action"] as? [String: Any] ?? parsed
  }

  private func executeAction(_ action: [String: Any]) throws -> ActionOutcome {
    switch action["type"] as? String {
    case "tap":
      guard let point = resolveTapPoint(action) else {
        return ActionOutcome(message: "Tap target could not be resolved: \(jsonString(action))", done: false, consumedAction: false)
      }
      let delivered = injectTap(point)
      return ActionOutcome(message: delivered ? "Tapped at \(Int(point.x)), \(Int(point.y))." : "No tappable control found at \(Int(point.x)), \(Int(point.y)).", done: false, consumedAction: true)
    case "wait":
      let ms = min(max(action["ms"] as? Int ?? 500, 0), 5_000)
      Self.sleep(milliseconds: ms)
      return ActionOutcome(message: "Waited \(ms)ms.", done: false, consumedAction: true)
    case "askUser":
      paused = true
      setState("paused")
      return ActionOutcome(message: action["question"] as? String ?? "Agent needs user input.", done: false, consumedAction: false)
    case "done":
      return ActionOutcome(message: action["reason"] as? String ?? "Task complete.", done: true, consumedAction: false)
    default:
      return ActionOutcome(message: "Unsupported action: \(action["type"] as? String ?? "unknown")", done: false, consumedAction: false)
    }
  }

  private func resolveTapPoint(_ action: [String: Any]) -> CGPoint? {
    if let x = action["x"] as? Double, let y = action["y"] as? Double {
      return CGPoint(x: x, y: y)
    }
    guard let nodeId = action["nodeId"] as? String, let rect = nodeRects[nodeId] else {
      return nil
    }
    return CGPoint(x: rect.midX, y: rect.midY)
  }

  private func injectTap(_ point: CGPoint) -> Bool {
    return (try? runOnMain {
      guard let window = Self.activeWindow() else {
        return false
      }
      guard let view = window.hitTest(point, with: nil) else {
        return false
      }
      if let control = view as? UIControl {
        control.sendActions(for: .touchUpInside)
        return true
      }
      view.gestureRecognizers?.forEach { recognizer in
        recognizer.isEnabled = recognizer.isEnabled
      }
      return false
    }) ?? false
  }

  private func snapshotBlocking() throws -> String {
    return try runOnMain {
      guard let window = Self.activeWindow() else {
        throw ZeticLLMError.invalidOption("NO_WINDOW: Cannot snapshot without an active window.")
      }
      var rects: [String: CGRect] = [:]
      var lines = ["iOS view hierarchy"]
      self.appendView(window, id: "0", depth: 0, lines: &lines, rects: &rects)
      self.nodeRects = rects
      return lines.joined(separator: "\n")
    }
  }

  private func appendView(
    _ view: UIView,
    id: String,
    depth: Int,
    lines: inout [String],
    rects: inout [String: CGRect]
  ) {
    guard !view.isHidden, view.alpha > 0.01 else { return }
    let rect = view.convert(view.bounds, to: nil)
    rects[id] = rect
    let indent = String(repeating: "  ", count: depth)
    var parts = [
      "\(indent)\(id)",
      String(describing: type(of: view)),
      "rect=(\(Int(rect.minX)),\(Int(rect.minY)),\(Int(rect.width)),\(Int(rect.height)))",
      "enabled=\((view as? UIControl)?.isEnabled ?? view.isUserInteractionEnabled)"
    ]
    if let label = view.accessibilityLabel, !label.isEmpty {
      parts.append("label=\"\(label.snapshotEscaped)\"")
    }
    if let label = view as? UILabel, let text = label.text, !text.isEmpty {
      parts.append("text=\"\(text.snapshotEscaped)\"")
    } else if let button = view as? UIButton, let text = button.title(for: .normal), !text.isEmpty {
      parts.append("text=\"\(text.snapshotEscaped)\"")
    } else if let textField = view as? UITextField {
      if let text = textField.text, !text.isEmpty {
        parts.append("value=\"\(text.snapshotEscaped)\"")
      }
      if let placeholder = textField.placeholder, !placeholder.isEmpty {
        parts.append("placeholder=\"\(placeholder.snapshotEscaped)\"")
      }
    }
    lines.append(parts.joined(separator: " "))

    for (index, child) in view.subviews.enumerated() {
      appendView(child, id: "\(id).\(index)", depth: depth + 1, lines: &lines, rects: &rects)
    }
  }

  private func waitWhilePaused() {
    while paused && !stopRequested {
      Self.sleep(milliseconds: 100)
    }
  }

  private func setState(_ next: String) {
    state = next
  }

  private func emit(
    type: String,
    message: String,
    snapshot: String? = nil,
    actionJson: String? = nil,
    progress: Double? = nil
  ) {
    NSLog("[ReactNativeZeticAgent] \(type): \(message)")
    listener?(
      AgentEvent(
        type: type,
        state: state,
        message: message,
        snapshot: snapshot,
        actionJson: actionJson,
        progress: progress
      )
    )
  }

  private func jsonString(_ value: [String: Any]) -> String {
    guard JSONSerialization.isValidJSONObject(value),
          let data = try? JSONSerialization.data(withJSONObject: value),
          let string = String(data: data, encoding: .utf8) else {
      return "{}"
    }
    return string
  }

  private func runOnMain<T>(_ block: @escaping () throws -> T) throws -> T {
    if Thread.isMainThread {
      return try block()
    }

    var result: Result<T, Error>!
    DispatchQueue.main.sync {
      do {
        result = .success(try block())
      } catch {
        result = .failure(error)
      }
    }
    return try result.get()
  }

  private static func activeWindow() -> UIWindow? {
    return UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }
      .flatMap { $0.windows }
      .first { $0.isKeyWindow }
  }

  private static func sleep(milliseconds: Int) {
    usleep(useconds_t(max(0, milliseconds) * 1_000))
  }

  private static func normalize(_ value: String?) -> String {
    return (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
  }

  private static func makeModelMode(_ value: String?) -> LLMModelMode {
    switch normalize(value) {
    case "RUN_SPEED": return .RUN_SPEED
    case "RUN_ACCURACY": return .RUN_ACCURACY
    default: return .RUN_AUTO
    }
  }

  private static func makeDataSetType(_ value: String?) throws -> LLMDataSetType? {
    switch normalize(value) {
    case "": return nil
    case "MMLU": return .MMLU
    case "TRUTHFULQA": return .TRUTHFULQA
    case "CNN_DAILYMAIL": return .CNN_DAILYMAIL
    case "GSM8K": return .GSM8K
    default: throw ZeticLLMError.invalidOption("Unsupported dataSetType: \(value ?? "")")
    }
  }

  private static func makeCachePolicy(_ value: String?) -> ZeticMLangeCacheHandlingPolicy {
    switch normalize(value) {
    case "KEEP_EXISTING": return .KEEP_EXISTING
    default: return .REMOVE_OVERLAPPING
    }
  }

  private static func makeKVPolicy(_ value: String?) -> LLMKVCacheCleanupPolicy {
    switch normalize(value) {
    case "DO_NOT_CLEAN_UP": return .DO_NOT_CLEAN_UP
    default: return .CLEAN_UP_ON_FULL
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
    case "LLAMA_CPP": return .LLAMA_CPP
    default: throw ZeticLLMError.invalidOption("Unsupported target: \(value)")
    }
  }

  private static func makeQuantType(_ value: String) throws -> LLMQuantType {
    switch normalize(value) {
    case "GGUF_QUANT_ORG": return .GGUF_QUANT_ORG
    case "GGUF_QUANT_F16": return .GGUF_QUANT_F16
    case "GGUF_QUANT_BF16": return .GGUF_QUANT_BF16
    case "GGUF_QUANT_Q8_0": return .GGUF_QUANT_Q8_0
    case "GGUF_QUANT_Q6_K": return .GGUF_QUANT_Q6_K
    case "GGUF_QUANT_Q4_K_M": return .GGUF_QUANT_Q4_K_M
    case "GGUF_QUANT_Q3_K_M": return .GGUF_QUANT_Q3_K_M
    case "GGUF_QUANT_Q2_K": return .GGUF_QUANT_Q2_K
    case "GGUF_QUANT_NUM_TYPES": return .GGUF_QUANT_NUM_TYPES
    default: throw ZeticLLMError.invalidOption("Unsupported quantType: \(value)")
    }
  }

  private static func makeAPType(_ value: String?) throws -> APType {
    switch normalize(value) {
    case "", "CPU": return .CPU
    case "GPU": return .GPU
    case "NPU": throw ZeticLLMError.invalidOption("APType.NPU is not supported by ZeticMLangeLLMModel on iOS.")
    default: throw ZeticLLMError.invalidOption("Unsupported apType: \(value ?? "")")
    }
  }
}

private extension String {
  var snapshotEscaped: String {
    String(prefix(240))
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
      .replacingOccurrences(of: "\n", with: " ")
  }
}

private extension Comparable {
  func clamped(to range: ClosedRange<Self>) -> Self {
    min(max(self, range.lowerBound), range.upperBound)
  }
}

import Foundation
import NitroModules
import UIKit
import ZeticMLange

final class HybridZeticLLMModel: HybridZeticLLMModelSpec {
  private let lock = NSLock()
  private var model: ZeticMLangeLLMModel?
  private var isGenerating = false
  private let config: NativeLoadModelConfig?

  init(model: ZeticMLangeLLMModel, config: NativeLoadModelConfig? = nil) {
    self.model = model
    self.config = config
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

  func generateMultimodal(
    config: NativeMultimodalGenerateConfig,
    onToken: ((TokenEvent) -> Void)?
  ) throws -> Promise<GenerateResult> {
    return Promise.async {
      let activeModel = try self.beginGeneration()
      defer { self.endGeneration() }

      try activeModel.validate(profile: config.profile.toMultimodalProfile())

      var mediaEmbeddings: [String: [Float]] = [:]
      for block in config.blocks {
        let type = block.type.lowercased()
        guard type == "audio" || type == "image" else {
          continue
        }
        guard let id = block.id else {
          throw ZeticLLMError.invalidOption("\(type) block is missing id.")
        }
        if mediaEmbeddings[id] != nil {
          continue
        }
        guard let input = block.input else {
          throw ZeticLLMError.invalidOption("\(type) block '\(id)' is missing input.")
        }

        if type == "audio" {
          mediaEmbeddings[id] = try self.encodeAudioBlock(
            input: input,
            encoderConfig: config.audioEncoder,
            audioPreprocess: config.audioPreprocess)
        } else {
          mediaEmbeddings[id] = try self.encodeImageBlock(
            input: input,
            encoderConfig: config.imageEncoder,
            imagePreprocess: config.imagePreprocess)
        }
      }

      let embeddings = try config.blocks.flatMap { block in
        try block.toEmbeddings(llm: activeModel, mediaEmbeddings: mediaEmbeddings)
      }
      let runResult = try activeModel.runWithEmbeddings(embeddings)
      return self.collectTokens(
        promptTokens: runResult.promptTokens,
        initialStatus: 0,
        activeModel: activeModel,
        onToken: onToken)
    }
  }

  func runWithEmbeddings(
    embeddings: ArrayBuffer,
    onToken: ((TokenEvent) -> Void)?
  ) throws -> Promise<GenerateResult> {
    let ownedEmbeddings = embeddings.asOwning()
    return Promise.async {
      let activeModel = try self.beginGeneration()
      defer { self.endGeneration() }

      let runResult = try activeModel.runWithEmbeddings(try ownedEmbeddings.toFloatArray())
      return self.collectTokens(
        promptTokens: runResult.promptTokens,
        initialStatus: 0,
        activeModel: activeModel,
        onToken: onToken)
    }
  }

  func tokenize(text: String, parseSpecial: Bool?) throws -> Promise<[Double]> {
    return Promise.async {
      guard let model = self.currentModel() else {
        throw ZeticLLMError.modelReleased
      }
      return try model.tokenize(text, parseSpecial: parseSpecial ?? false).map(Double.init)
    }
  }

  func tokenEmbeddings(tokenIds: [Double]) throws -> Promise<ArrayBuffer> {
    return Promise.async {
      guard let model = self.currentModel() else {
        throw ZeticLLMError.modelReleased
      }
      let ids = tokenIds.map { Int32($0) }
      return try model.tokenEmbeddings(ids).toArrayBuffer()
    }
  }

  func specialTokenId(name: String) throws -> Promise<Double> {
    return Promise.async {
      guard let model = self.currentModel() else {
        throw ZeticLLMError.modelReleased
      }
      return Double(try model.specialTokenId(name))
    }
  }

  func validateMultimodalProfile(profile: NativeMultimodalProfile) throws -> Promise<Void> {
    return Promise.async {
      guard let model = self.currentModel() else {
        throw ZeticLLMError.modelReleased
      }
      try model.validate(profile: profile.toMultimodalProfile())
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

  private func collectTokens(
    promptTokens: Int,
    initialStatus: Int,
    activeModel: ZeticMLangeLLMModel,
    onToken: ((TokenEvent) -> Void)?
  ) -> GenerateResult {
    var output = ""
    var generatedTokens = 0
    var finalStatus = initialStatus

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
      promptTokens: Double(promptTokens),
      generatedTokens: Double(generatedTokens),
      status: Double(finalStatus)
    )
  }

  private func encodeAudioBlock(
    input: NativeMediaInput,
    encoderConfig: NativeMultimodalEncoderConfig?,
    audioPreprocess: String?
  ) throws -> [Float] {
    guard let encoderConfig else {
      throw ZeticLLMError.invalidOption("audioEncoder is required for audio blocks.")
    }
    guard (audioPreprocess ?? "qwen-omni-audio") == "qwen-omni-audio" else {
      throw ZeticLLMError.invalidOption("Unsupported audioPreprocess: \(audioPreprocess ?? "")")
    }
    guard encoderConfig.inputDataType.lowercased() == "float32" else {
      throw ZeticLLMError.invalidOption("qwen-omni-audio requires a float32 encoder input.")
    }

    let pcm = try input.toPcmFloatArray().resampleMonoTo16k()
    let chunks = QwenOmniAudioPreprocessor.melChunks(samples: pcm.samples)
    return try runEncoderChunks(config: encoderConfig, chunks: chunks)
  }

  private func encodeImageBlock(
    input: NativeMediaInput,
    encoderConfig: NativeMultimodalEncoderConfig?,
    imagePreprocess: NativeImagePreprocessConfig?
  ) throws -> [Float] {
    guard let encoderConfig else {
      throw ZeticLLMError.invalidOption("imageEncoder is required for image blocks.")
    }
    guard let imagePreprocess else {
      throw ZeticLLMError.invalidOption("imagePreprocess is required for image blocks.")
    }
    let tensor = try input.toImageTensor(preprocess: imagePreprocess, encoder: encoderConfig)
    return try runEncoderOnce(config: encoderConfig, payload: tensor)
  }

  private func runEncoderChunks(
    config: NativeMultimodalEncoderConfig,
    chunks: [TensorPayload]
  ) throws -> [Float] {
    let encoder = try createEncoder(config.model)
    var embeddings: [Float] = []
    for payload in chunks {
      let outputs = try encoder.run(inputs: [payload.toTensor()])
      let output = outputs[config.outputIndexValue()]
      embeddings.append(contentsOf: try output.toFloatArray())
    }
    return embeddings
  }

  private func runEncoderOnce(
    config: NativeMultimodalEncoderConfig,
    payload: TensorPayload
  ) throws -> [Float] {
    let encoder = try createEncoder(config.model)
    let outputs = try encoder.run(inputs: [payload.toTensor()])
    return try outputs[config.outputIndexValue()].toFloatArray()
  }

  private func createEncoder(_ config: NativeLoadModelConfig) throws -> ZeticMLangeModel {
    if config.explicitRuntime != nil {
      throw ZeticLLMError.invalidOption("explicitRuntime is not supported for encoder models yet.")
    }
    return try ZeticMLangeModel(
      personalKey: config.personalKey,
      name: config.name,
      version: config.version.map { Int($0) },
      modelMode: Self.makeGenericModelMode(config.modelMode),
      cacheHandlingPolicy: Self.makeCachePolicy(config.cacheHandlingPolicy)
    )
  }

  private static func makeGenericModelMode(_ value: String?) -> ModelMode {
    switch (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased() {
    case "RUN_SPEED":
      return .RUN_SPEED
    case "RUN_ACCURACY":
      return .RUN_ACCURACY
    default:
      return .RUN_AUTO
    }
  }

  private static func makeCachePolicy(_ value: String?) -> ZeticMLangeCacheHandlingPolicy {
    switch (value ?? "").trimmingCharacters(in: .whitespacesAndNewlines).uppercased() {
    case "KEEP_EXISTING":
      return .KEEP_EXISTING
    default:
      return .REMOVE_OVERLAPPING
    }
  }
}

private extension NativePromptEmbeddingBlock {
  func toEmbeddings(
    llm: ZeticMLangeLLMModel,
    mediaEmbeddings: [String: [Float]]
  ) throws -> [Float] {
    switch type.lowercased() {
    case "text":
      guard let text else {
        throw ZeticLLMError.invalidOption("text block is missing text.")
      }
      let ids = try llm.tokenize(text, parseSpecial: parseSpecial ?? false)
      return try llm.tokenEmbeddings(ids)
    case "audio", "image":
      guard let id else {
        throw ZeticLLMError.invalidOption("media block is missing id.")
      }
      guard let embeddings = mediaEmbeddings[id] else {
        throw ZeticLLMError.invalidOption("No embeddings were produced for '\(id)'.")
      }
      return embeddings
    default:
      throw ZeticLLMError.invalidOption("Unsupported block type: \(type)")
    }
  }
}

private extension NativeMultimodalEncoderConfig {
  func outputIndexValue() -> Int {
    return outputIndex.map { Int($0) } ?? 0
  }
}

private extension NativeMultimodalProfile {
  func toMultimodalProfile() -> MultimodalProfile {
    return MultimodalProfile(
      name: name,
      requiredSpecialTokens: requiredSpecialTokens
    )
  }
}

private extension NativeMediaInput {
  func toPcmFloatArray() throws -> PcmAudio {
    switch type.lowercased() {
    case "pcm":
      return try decodePcmBuffer()
    case "bytes":
      guard let data else {
        throw ZeticLLMError.invalidOption("bytes audio input is missing data.")
      }
      return try parseWavBytes([UInt8](data.toData(copyIfNeeded: true)))
    case "uri":
      guard let uri else {
        throw ZeticLLMError.invalidOption("uri audio input is missing uri.")
      }
      return try parseWavBytes([UInt8](readUriData(uri)))
    default:
      throw ZeticLLMError.invalidOption("audio blocks accept uri, bytes, or pcm inputs.")
    }
  }

  func decodePcmBuffer() throws -> PcmAudio {
    guard let sampleRate else {
      throw ZeticLLMError.invalidOption("pcm input is missing sampleRate.")
    }
    guard let channels else {
      throw ZeticLLMError.invalidOption("pcm input is missing channels.")
    }
    guard let format else {
      throw ZeticLLMError.invalidOption("pcm input is missing format.")
    }
    guard let data else {
      throw ZeticLLMError.invalidOption("pcm input is missing data.")
    }
    let bytes = data.toData(copyIfNeeded: true)
    let samples: [Float]
    switch format.lowercased() {
    case "float32":
      if bytes.count % MemoryLayout<Float>.size != 0 {
        throw ZeticLLMError.invalidOption("float32 pcm byteLength must be divisible by 4.")
      }
      samples = bytes.withUnsafeBytes { raw in
        Array(raw.bindMemory(to: Float.self))
      }
    case "int16":
      if bytes.count % MemoryLayout<Int16>.size != 0 {
        throw ZeticLLMError.invalidOption("int16 pcm byteLength must be divisible by 2.")
      }
      samples = bytes.withUnsafeBytes { raw in
        raw.bindMemory(to: Int16.self).map { Float($0) / Float(Int16.max) }
      }
    default:
      throw ZeticLLMError.invalidOption("Unsupported pcm format: \(format)")
    }
    return PcmAudio(samples: samples, sampleRate: Int(sampleRate), channels: Int(channels))
  }

  func toImageTensor(
    preprocess: NativeImagePreprocessConfig,
    encoder: NativeMultimodalEncoderConfig
  ) throws -> TensorPayload {
    let targetWidth = Int(preprocess.width)
    let targetHeight = Int(preprocess.height)
    let image = try toUIImage().resized(
      width: targetWidth,
      height: targetHeight,
      mode: preprocess.resizeMode ?? "stretch")
    let rgba = try image.rgbaBytes(width: targetWidth, height: targetHeight)
    let shape = encoder.inputShape.map { Int($0) }
    let order = (preprocess.colorOrder ?? "rgb").lowercased() == "bgr" ? [2, 1, 0] : [0, 1, 2]
    let layout = (preprocess.layout ?? "nchw").lowercased()
    let mean = (preprocess.mean ?? [0, 0, 0]).map(Float.init)
    let std = (preprocess.stdValues ?? [1, 1, 1]).map(Float.init)
    let pixelCount = targetWidth * targetHeight

    switch encoder.inputDataType.lowercased() {
    case "float32":
      var floats = [Float](repeating: 0, count: pixelCount * 3)
      for y in 0..<targetHeight {
        for x in 0..<targetWidth {
          let pixelIndex = y * targetWidth + x
          let base = pixelIndex * 4
          let values = [
            Float(rgba[base]) / 255.0,
            Float(rgba[base + 1]) / 255.0,
            Float(rgba[base + 2]) / 255.0,
          ]
          for channel in 0..<3 {
            let value = (values[order[channel]] - mean[safe: channel, default: 0]) / std[safe: channel, default: 1]
            let outIndex = layout == "nhwc" ? pixelIndex * 3 + channel : channel * pixelCount + pixelIndex
            floats[outIndex] = value
          }
        }
      }
      return .float32(floats, shape: shape)
    case "uint8", "int8":
      var bytes = [UInt8](repeating: 0, count: pixelCount * 3)
      for y in 0..<targetHeight {
        for x in 0..<targetWidth {
          let pixelIndex = y * targetWidth + x
          let base = pixelIndex * 4
          let values = [rgba[base], rgba[base + 1], rgba[base + 2]]
          for channel in 0..<3 {
            let value = values[order[channel]]
            let outIndex = layout == "nhwc" ? pixelIndex * 3 + channel : channel * pixelCount + pixelIndex
            bytes[outIndex] = encoder.inputDataType.lowercased() == "int8"
              ? UInt8(bitPattern: Int8(Int(value) - 128))
              : value
          }
        }
      }
      return encoder.inputDataType.lowercased() == "int8"
        ? .int8(bytes, shape: shape)
        : .uint8(bytes, shape: shape)
    default:
      throw ZeticLLMError.invalidOption("Unsupported encoder inputDataType: \(encoder.inputDataType)")
    }
  }

  func toUIImage() throws -> UIImage {
    switch type.lowercased() {
    case "bytes":
      guard let data else {
        throw ZeticLLMError.invalidOption("bytes image input is missing data.")
      }
      guard let image = UIImage(data: data.toData(copyIfNeeded: true)) else {
        throw ZeticLLMError.invalidOption("Could not decode image bytes.")
      }
      return image
    case "uri":
      guard let uri else {
        throw ZeticLLMError.invalidOption("uri image input is missing uri.")
      }
      guard let image = UIImage(data: try readUriData(uri)) else {
        throw ZeticLLMError.invalidOption("Could not decode image uri: \(uri)")
      }
      return image
    case "pixels":
      return try pixelsToUIImage()
    default:
      throw ZeticLLMError.invalidOption("image blocks accept uri, bytes, or pixels inputs.")
    }
  }

  func pixelsToUIImage() throws -> UIImage {
    guard let data else {
      throw ZeticLLMError.invalidOption("pixels image input is missing data.")
    }
    guard let width, let height else {
      throw ZeticLLMError.invalidOption("pixels image input is missing width/height.")
    }
    guard let format else {
      throw ZeticLLMError.invalidOption("pixels image input is missing format.")
    }
    let inputBytes = [UInt8](data.toData(copyIfNeeded: true))
    let pixelCount = Int(width) * Int(height)
    let stride: Int
    switch format.lowercased() {
    case "rgba8":
      stride = 4
    case "rgb8":
      stride = 3
    default:
      throw ZeticLLMError.invalidOption("Unsupported pixel format: \(format)")
    }
    if inputBytes.count < pixelCount * stride {
      throw ZeticLLMError.invalidOption("pixels buffer is smaller than width * height * channels.")
    }

    var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
    for index in 0..<pixelCount {
      let source = index * stride
      let target = index * 4
      rgba[target] = inputBytes[source]
      rgba[target + 1] = inputBytes[source + 1]
      rgba[target + 2] = inputBytes[source + 2]
      rgba[target + 3] = stride == 4 ? inputBytes[source + 3] : 255
    }

    let data = Data(rgba) as CFData
    guard
      let provider = CGDataProvider(data: data),
      let cgImage = CGImage(
        width: Int(width),
        height: Int(height),
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: Int(width) * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent)
    else {
      throw ZeticLLMError.invalidOption("Could not create image from pixels.")
    }
    return UIImage(cgImage: cgImage)
  }
}

private enum TensorPayload {
  case float32([Float], shape: [Int])
  case uint8([UInt8], shape: [Int])
  case int8([UInt8], shape: [Int])

  func toTensor() -> Tensor {
    switch self {
    case .float32(let values, let shape):
      let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
      return Tensor(data: data, dataType: BuiltinDataType.float32, shape: shape)
    case .uint8(let values, let shape):
      return Tensor(data: Data(values), dataType: BuiltinDataType.uint8, shape: shape)
    case .int8(let values, let shape):
      return Tensor(data: Data(values), dataType: BuiltinDataType.int8, shape: shape)
    }
  }
}

private struct PcmAudio {
  let samples: [Float]
  let sampleRate: Int
  let channels: Int

  func resampleMonoTo16k() -> PcmAudio {
    var mono: [Float]
    if channels == 1 {
      mono = samples
    } else {
      mono = []
      mono.reserveCapacity(samples.count / channels)
      for frame in 0..<(samples.count / channels) {
        var sum: Float = 0
        for channel in 0..<channels {
          sum += samples[frame * channels + channel]
        }
        mono.append(sum / Float(channels))
      }
    }

    if sampleRate == 16_000 {
      return PcmAudio(samples: mono, sampleRate: 16_000, channels: 1)
    }

    let ratio = Double(sampleRate) / 16_000.0
    let outputSize = max(1, Int(floor(Double(mono.count) / ratio)))
    let resampled = (0..<outputSize).map { index -> Float in
      let source = Double(index) * ratio
      let left = min(max(Int(floor(source)), 0), mono.count - 1)
      let right = min(left + 1, mono.count - 1)
      let fraction = Float(source - Double(left))
      return mono[left] * (1 - fraction) + mono[right] * fraction
    }
    return PcmAudio(samples: resampled, sampleRate: 16_000, channels: 1)
  }
}

private enum QwenOmniAudioPreprocessor {
  static let sampleRate = 16_000
  static let chunkSamples = sampleRate * 2
  static let nFft = 400
  static let hopLength = 160
  static let melBins = 128
  static let framesPerChunk = 200
  static let window = (0..<nFft).map { index in
    Float(0.5 - 0.5 * cos((2.0 * Double.pi * Double(index)) / Double(nFft)))
  }
  static let melFilter = createMelFilter()

  static func melChunks(samples: [Float]) -> [TensorPayload] {
    let totalChunks = max(1, (samples.count + chunkSamples - 1) / chunkSamples)
    return (0..<totalChunks).map { chunkIndex in
      let offset = chunkIndex * chunkSamples
      let chunk = (0..<chunkSamples).map { index -> Float in
        let sampleIndex = offset + index
        return sampleIndex < samples.count ? samples[sampleIndex] : 0
      }
      return .float32(logMel(samples: chunk), shape: [1, melBins, framesPerChunk])
    }
  }

  static func logMel(samples: [Float]) -> [Float] {
    var output = [Float](repeating: 0, count: melBins * framesPerChunk)
    var frame = [Float](repeating: 0, count: nFft)
    for frameIndex in 0..<framesPerChunk {
      let start = frameIndex * hopLength - nFft / 2
      for i in 0..<nFft {
        let sampleIndex = start + i
        frame[i] = (sampleIndex >= 0 && sampleIndex < samples.count ? samples[sampleIndex] : 0) * window[i]
      }
      let spectrum = powerSpectrum(frame: frame)
      for mel in 0..<melBins {
        var energy: Float = 0
        for bin in spectrum.indices {
          energy += spectrum[bin] * melFilter[mel][bin]
        }
        let logValue = Float(log10(Double(max(energy, 1.0e-10))))
        output[mel * framesPerChunk + frameIndex] = min(max((logValue + 4) / 4, -1), 2)
      }
    }
    return output
  }

  static func powerSpectrum(frame: [Float]) -> [Float] {
    (0...(nFft / 2)).map { k in
      var real = 0.0
      var imaginary = 0.0
      for n in frame.indices {
        let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(nFft)
        real += Double(frame[n]) * cos(angle)
        imaginary -= Double(frame[n]) * sin(angle)
      }
      return Float((real * real + imaginary * imaginary) / Double(nFft))
    }
  }

  static func createMelFilter() -> [[Float]] {
    let melMin = hzToMel(0)
    let melMax = hzToMel(Double(sampleRate) / 2)
    let points = (0..<(melBins + 2)).map { index in
      melToHz(melMin + (melMax - melMin) * Double(index) / Double(melBins + 1))
    }
    let bins = points.map { hz in
      min(max(Int(floor(Double(nFft + 1) * hz / Double(sampleRate))), 0), nFft / 2)
    }
    return (0..<melBins).map { mel in
      (0...(nFft / 2)).map { bin in
        let left = bins[mel]
        let center = bins[mel + 1]
        let right = bins[mel + 2]
        if bin < left || bin > right {
          return 0
        }
        if bin <= center {
          return center == left ? 0 : Float(bin - left) / Float(center - left)
        }
        return right == center ? 0 : Float(right - bin) / Float(right - center)
      }
    }
  }

  static func hzToMel(_ hz: Double) -> Double {
    2595.0 * log10(1.0 + hz / 700.0)
  }

  static func melToHz(_ mel: Double) -> Double {
    700.0 * (pow(10.0, mel / 2595.0) - 1.0)
  }
}

private extension ArrayBuffer {
  func toFloatArray() throws -> [Float] {
    let data = toData(copyIfNeeded: true)
    if data.count % MemoryLayout<Float>.size != 0 {
      throw ZeticLLMError.invalidOption("embeddings byteLength must be divisible by 4.")
    }
    return data.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self))
    }
  }
}

private extension Array where Element == Float {
  func toArrayBuffer() throws -> ArrayBuffer {
    let data = withUnsafeBufferPointer { Data(buffer: $0) }
    return try ArrayBuffer.copy(data: data)
  }
}

private extension Tensor {
  func toFloatArray() throws -> [Float] {
    data.withUnsafeBytes { raw in
      Array(raw.bindMemory(to: Float.self).prefix(count()))
    }
  }
}

private extension UIImage {
  func resized(width: Int, height: Int, mode: String) -> UIImage {
    let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
    return renderer.image { context in
      UIColor.black.setFill()
      context.fill(CGRect(x: 0, y: 0, width: width, height: height))

      let sourceSize = size
      let targetRect: CGRect
      switch mode.lowercased() {
      case "contain":
        let scale = min(CGFloat(width) / sourceSize.width, CGFloat(height) / sourceSize.height)
        let scaled = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        targetRect = CGRect(
          x: (CGFloat(width) - scaled.width) / 2,
          y: (CGFloat(height) - scaled.height) / 2,
          width: scaled.width,
          height: scaled.height)
      case "cover":
        let scale = max(CGFloat(width) / sourceSize.width, CGFloat(height) / sourceSize.height)
        let scaled = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        targetRect = CGRect(
          x: (CGFloat(width) - scaled.width) / 2,
          y: (CGFloat(height) - scaled.height) / 2,
          width: scaled.width,
          height: scaled.height)
      default:
        targetRect = CGRect(x: 0, y: 0, width: width, height: height)
      }
      draw(in: targetRect)
    }
  }

  func rgbaBytes(width: Int, height: Int) throws -> [UInt8] {
    guard let cgImage else {
      throw ZeticLLMError.invalidOption("Could not read image pixels.")
    }
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    let didDraw = bytes.withUnsafeMutableBytes { pointer -> Bool in
      guard let baseAddress = pointer.baseAddress else {
        return false
      }
      guard let context = CGContext(
        data: baseAddress,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
      else {
        return false
      }
      context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
      return true
    }
    if !didDraw {
      throw ZeticLLMError.invalidOption("Could not create image pixel buffer.")
    }
    return bytes
  }
}

private extension Array {
  subscript(safe index: Int, default defaultValue: Element) -> Element {
    indices.contains(index) ? self[index] : defaultValue
  }
}

private func readUriData(_ uri: String) throws -> Data {
  let url: URL
  if uri.hasPrefix("file://") {
    guard let parsed = URL(string: uri) else {
      throw ZeticLLMError.invalidOption("Invalid file uri: \(uri)")
    }
    url = parsed
  } else {
    url = URL(fileURLWithPath: uri)
  }
  return try Data(contentsOf: url)
}

private func parseWavBytes(_ bytes: [UInt8]) throws -> PcmAudio {
  guard bytes.count >= 44, bytes.readAscii(0, 4) == "RIFF", bytes.readAscii(8, 4) == "WAVE" else {
    throw ZeticLLMError.invalidOption("uri/bytes audio currently supports PCM WAV data.")
  }

  var offset = 12
  var audioFormat = 0
  var channels = 0
  var sampleRate = 0
  var bitsPerSample = 0
  var dataOffset = -1
  var dataSize = 0

  while offset + 8 <= bytes.count {
    let chunkId = bytes.readAscii(offset, 4)
    let chunkSize = bytes.readIntLe(offset + 4)
    let payload = offset + 8
    if payload + chunkSize > bytes.count { break }

    if chunkId == "fmt " {
      audioFormat = bytes.readShortLe(payload)
      channels = bytes.readShortLe(payload + 2)
      sampleRate = bytes.readIntLe(payload + 4)
      bitsPerSample = bytes.readShortLe(payload + 14)
    } else if chunkId == "data" {
      dataOffset = payload
      dataSize = chunkSize
    }

    offset = payload + chunkSize + (chunkSize % 2)
  }

  guard dataOffset >= 0, channels > 0, sampleRate > 0 else {
    throw ZeticLLMError.invalidOption("WAV data is missing fmt or data chunks.")
  }

  let samples: [Float]
  if audioFormat == 1 && bitsPerSample == 16 {
    samples = (0..<(dataSize / 2)).map { index in
      Float(Int16(bitPattern: UInt16(bytes.readShortLe(dataOffset + index * 2)))) / Float(Int16.max)
    }
  } else if audioFormat == 3 && bitsPerSample == 32 {
    samples = (0..<(dataSize / 4)).map { index in
      Float(bitPattern: bytes.readUInt32Le(dataOffset + index * 4))
    }
  } else {
    throw ZeticLLMError.invalidOption("WAV must be 16-bit PCM or 32-bit float.")
  }
  return PcmAudio(samples: samples, sampleRate: sampleRate, channels: channels)
}

private extension Array where Element == UInt8 {
  func readAscii(_ offset: Int, _ length: Int) -> String {
    String(bytes: self[offset..<(offset + length)], encoding: .ascii) ?? ""
  }

  func readIntLe(_ offset: Int) -> Int {
    Int(self[offset]) |
      (Int(self[offset + 1]) << 8) |
      (Int(self[offset + 2]) << 16) |
      (Int(self[offset + 3]) << 24)
  }

  func readUInt32Le(_ offset: Int) -> UInt32 {
    UInt32(self[offset]) |
      (UInt32(self[offset + 1]) << 8) |
      (UInt32(self[offset + 2]) << 16) |
      (UInt32(self[offset + 3]) << 24)
  }

  func readShortLe(_ offset: Int) -> Int {
    Int(self[offset]) | (Int(self[offset + 1]) << 8)
  }
}

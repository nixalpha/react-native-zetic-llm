# react-native-zetic-llm

React Native Nitro module for running Zetic Melange LLMs locally on iOS and Android.

## Installation

```sh
npm install react-native-zetic-llm react-native-nitro-modules
npx pod-install
```

Then generate the Nitro bindings:

```sh
npx nitrogen
```

## iOS Setup

ZeticMLangeiOS is added as a Swift Package from the consuming app's Podfile. Require the helper and call it from the existing `post_install` block:

```ruby
require File.join(File.dirname(`node --print "require.resolve('react-native-zetic-llm/package.json')"`), "ios", "zetic_mlange_spm")

post_install do |installer|
  react_native_post_install(installer, config[:reactNativePath])
  react_native_zetic_llm_post_install(installer)
end
```

The helper pins `https://github.com/zetic-ai/ZeticMLangeiOS.git` to exact version `1.7.0-beta.1`, links the `ZeticMLange` product, and keeps the dependency in sync on each `pod install`. ZeticMLangeiOS requires iOS 16.0, so the consuming app must use an iOS deployment target of 16.0 or newer.

## Android Setup

The library adds:

```gradle
implementation("com.zeticai.mlange:mlange:+")
```

The Android module also sets:

```gradle
packagingOptions {
  jniLibs {
    useLegacyPackaging true
  }
}
```

Use `compileSdkVersion >= 34`, `minSdkVersion >= 24`, and NDK 27 or newer.

## Usage

```ts
import { loadModel } from 'react-native-zetic-llm'

const model = await loadModel(
  {
    personalKey: 'dev_...',
    name: 'changgeun/gemma-4-E2B-it',
    version: 1,
    modelMode: 'RUN_AUTO',
  },
  (progress) => {
    console.log('download progress', progress)
  }
)

const result = await model.generate('prompt', ({ token }) => {
  console.log(token)
})

console.log(result.text)

await model.cleanUp()
model.release()
```

### Multimodal Embedding Injection

Multimodal support uses ZeticMLange `1.7.0-beta.1` and the SDK's `runWithEmbeddings` path. The low-level APIs are available on the loaded model:

```ts
const ids = await model.tokenize('<|im_start|>user\n', true)
const textEmbeddings = await model.tokenEmbeddings(ids)
const tokenId = await model.specialTokenId('<|audio_bos|>')

await model.validateMultimodalProfile({
  name: 'qwen-omni-audio',
  requiredSpecialTokens: ['<|audio_bos|>', '<|audio_eos|>', '<|im_start|>', '<|im_end|>'],
})

const result = await model.runWithEmbeddings(textEmbeddings, ({ token }) => {
  console.log(token)
})
```

For full audio/image flows, provide app-owned media inputs and encoder configs. Audio supports PCM buffers directly and PCM WAV `uri`/`bytes` inputs; image supports decoded `uri`/`bytes` inputs and raw `rgb8`/`rgba8` pixel buffers.

```ts
const result = await model.generateMultimodal(
  {
    profile: {
      name: 'qwen-omni-audio',
      requiredSpecialTokens: ['<|audio_bos|>', '<|audio_eos|>', '<|im_start|>', '<|im_end|>'],
    },
    audioEncoder: {
      model: {
        personalKey: 'dev_...',
        name: 'zetic/qwen2.5_omni_audio_encoder_chunk_f16',
        modelMode: 'RUN_AUTO',
      },
      inputShape: [1, 128, 200],
      inputDataType: 'float32',
    },
    audioPreprocess: 'qwen-omni-audio',
    blocks: [
      { type: 'text', text: '<|im_start|>user\n<|audio_bos|>', parseSpecial: true },
      {
        type: 'audio',
        id: 'clip',
        input: {
          type: 'pcm',
          data: pcmFloat32.buffer,
          sampleRate: 16000,
          channels: 1,
          format: 'float32',
        },
      },
      { type: 'text', text: '<|audio_eos|>What do you hear?<|im_end|>\n<|im_start|>assistant\n', parseSpecial: true },
    ],
  },
  ({ token }) => console.log(token)
)
```

## API

```ts
type LLMModelMode = 'RUN_AUTO' | 'RUN_SPEED' | 'RUN_ACCURACY'
type LLMTarget = 'LLAMA_CPP'
type LLMDataSetType = 'MMLU' | 'TRUTHFULQA' | 'CNN_DAILYMAIL' | 'GSM8K'
type APType = 'CPU' | 'GPU' | 'NPU'
type LLMKVCacheCleanupPolicy = 'CLEAN_UP_ON_FULL' | 'DO_NOT_CLEAN_UP'
type CacheHandlingPolicy = 'REMOVE_OVERLAPPING' | 'KEEP_EXISTING'

interface LoadModelConfig {
  personalKey: string
  name: string
  version?: number
  modelMode?: LLMModelMode
  dataSetType?: LLMDataSetType
  cacheHandlingPolicy?: CacheHandlingPolicy
  initOption?: {
    kvCacheCleanupPolicy?: LLMKVCacheCleanupPolicy
    nCtx?: number
  }
  explicitRuntime?: {
    target: LLMTarget
    quantType: LLMQuantType
    apType?: APType
  }
}
```

## Notes

- Generation is single-flight per model instance. A second concurrent `generate()` call rejects with `GENERATION_IN_PROGRESS`.
- `release()` is idempotent and frees the native model. Future generation/cleanup calls reject with `MODEL_RELEASED`.
- The convenience `loadModel()` validates configuration before crossing the native boundary.
- Sampling options are intentionally not exposed because the referenced Zetic APIs document prompt execution and token consumption only.
- Multimodal embedding injection is beta in ZeticMLange `1.7.0-beta.1` and is currently supported by Zetic only on the llama.cpp LLM backend.

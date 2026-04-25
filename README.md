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

The helper pins `https://github.com/zetic-ai/ZeticMLangeiOS.git` to exact version `1.6.0`, links the `ZeticMLange` product, and keeps the dependency in sync on each `pod install`. ZeticMLangeiOS `1.6.0` requires iOS 16.0, so the consuming app must use an iOS deployment target of 16.0 or newer.

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

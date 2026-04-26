import { preloadModel } from '../index'
import {
  validateLoadModelConfig,
  validateMultimodalGenerateConfig,
  validateMultimodalProfile,
  type ModelProgressEvent,
} from '../validation'

describe('validateLoadModelConfig', () => {
  it('accepts automatic model configuration', () => {
    expect(
      validateLoadModelConfig({
        personalKey: 'dev_key',
        name: 'google/gemma-3-4b-it',
        modelMode: 'RUN_AUTO',
        initOption: {
          kvCacheCleanupPolicy: 'CLEAN_UP_ON_FULL',
          nCtx: 4096,
        },
      })
    ).toEqual({
      personalKey: 'dev_key',
      name: 'google/gemma-3-4b-it',
      modelMode: 'RUN_AUTO',
      initOption: {
        kvCacheCleanupPolicy: 'CLEAN_UP_ON_FULL',
        nCtx: 4096,
      },
    })
  })

  it('rejects invalid explicit runtime values', () => {
    expect(() =>
      validateLoadModelConfig({
        personalKey: 'dev_key',
        name: 'model',
        explicitRuntime: {
          target: 'LLAMA_CPP',
          quantType: 'BAD' as never,
        },
      })
    ).toThrow(/explicitRuntime\.quantType/)
  })
})

describe('validateMultimodalProfile', () => {
  it('accepts custom multimodal profiles', () => {
    expect(
      validateMultimodalProfile({
        name: 'vision-model',
        requiredSpecialTokens: ['<|image_bos|>', '<|image_eos|>'],
      })
    ).toEqual({
      name: 'vision-model',
      requiredSpecialTokens: ['<|image_bos|>', '<|image_eos|>'],
    })
  })

  it('rejects empty special token names', () => {
    expect(() =>
      validateMultimodalProfile({
        name: 'broken',
        requiredSpecialTokens: [''],
      })
    ).toThrow(/requiredSpecialTokens\[0\]/)
  })
})

describe('validateMultimodalGenerateConfig', () => {
  const encoder = {
    model: {
      personalKey: 'dev_key',
      name: 'zetic/qwen2.5_omni_audio_encoder_chunk_f16',
      modelMode: 'RUN_AUTO' as const,
    },
    inputShape: [1, 128, 200],
    inputDataType: 'float32' as const,
  }

  it('accepts text and audio blocks with ArrayBuffer media', () => {
    const data = new ArrayBuffer(16)
    const config = {
      profile: {
        name: 'qwen-omni-audio',
        requiredSpecialTokens: ['<|audio_bos|>', '<|audio_eos|>'],
      },
      audioEncoder: encoder,
      audioPreprocess: 'qwen-omni-audio' as const,
      blocks: [
        { type: 'text' as const, text: '<|audio_bos|>', parseSpecial: true },
        {
          type: 'audio' as const,
          id: 'clip',
          input: {
            type: 'pcm' as const,
            data,
            sampleRate: 16000,
            channels: 1,
            format: 'float32' as const,
          },
        },
        { type: 'text' as const, text: '<|audio_eos|>', parseSpecial: true },
      ],
    }

    expect(validateMultimodalGenerateConfig(config)).toBe(config)
  })

  it('requires image preprocessing for image blocks', () => {
    expect(() =>
      validateMultimodalGenerateConfig({
        profile: {
          name: 'vision',
          requiredSpecialTokens: ['<|image_bos|>'],
        },
        imageEncoder: {
          model: {
            personalKey: 'dev_key',
            name: 'vision/encoder',
          },
          inputShape: [1, 3, 224, 224],
          inputDataType: 'float32',
        },
        blocks: [
          {
            type: 'image',
            id: 'image',
            input: {
              type: 'pixels',
              data: new ArrayBuffer(224 * 224 * 4),
              width: 224,
              height: 224,
              format: 'rgba8',
            },
          },
        ],
      })
    ).toThrow(/imagePreprocess/)
  })
})

describe('public api exports', () => {
  it('exposes preloadModel for auxiliary encoder downloads', () => {
    expect(typeof preloadModel).toBe('function')
  })

  it('supports structured model progress events', () => {
    const event: ModelProgressEvent = {
      phase: 'downloading',
      modelRole: 'imageEncoder',
      modelName: 'thetangylemon/qwen2_5_omni_vision_encoder_chunk_f16',
      progress: 0.42,
    }

    expect(event.phase).toBe('downloading')
    expect(event.modelRole).toBe('imageEncoder')
    expect(event.progress).toBe(0.42)
  })
})

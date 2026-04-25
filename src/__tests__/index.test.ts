import { validateLoadModelConfig } from '../validation'

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

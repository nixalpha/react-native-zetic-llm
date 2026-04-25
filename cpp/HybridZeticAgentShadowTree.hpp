#pragma once

#include "HybridZeticAgentShadowTreeSpec.hpp"

#include <jsi/jsi.h>
#include <react/renderer/components/root/RootShadowNode.h>
#include <react/renderer/uimanager/UIManager.h>
#include <react/renderer/uimanager/UIManagerMountHook.h>

#include <memory>
#include <mutex>
#include <string>

namespace margelo::nitro::zeticllm {

class HybridZeticAgentShadowTree final : public HybridZeticAgentShadowTreeSpec {
 public:
  HybridZeticAgentShadowTree();
  ~HybridZeticAgentShadowTree() override;

  std::string getLatestSnapshot() override;
  void setFallbackSnapshot(const std::string& snapshot) override;
  void clear() override;
  void dispose() override;

 protected:
  void loadHybridMethods() override;

 private:
  class MountObserver final : public facebook::react::UIManagerMountHook {
   public:
    explicit MountObserver(HybridZeticAgentShadowTree& owner);

    void shadowTreeDidMount(
        const facebook::react::RootShadowNode::Shared& rootShadowNode,
        facebook::react::HighResTimeStamp mountTime) noexcept override;

    void shadowTreeDidUnmount(
        facebook::react::SurfaceId surfaceId,
        facebook::react::HighResTimeStamp unmountTime) noexcept override;

   private:
    HybridZeticAgentShadowTree& owner_;
  };

  facebook::jsi::Value installShadowTreeObserverRaw(
      facebook::jsi::Runtime& runtime,
      const facebook::jsi::Value& thisValue,
      const facebook::jsi::Value* args,
      size_t count);

  void captureRoot(const facebook::react::RootShadowNode::Shared& rootShadowNode) noexcept;
  void unregisterObserver() noexcept;

  std::mutex mutex_;
  std::string latestSnapshot_;
  std::string fallbackSnapshot_;
  std::unique_ptr<MountObserver> observer_;
  facebook::react::UIManager* uiManager_{nullptr};
  bool registered_{false};
};

} // namespace margelo::nitro::zeticllm

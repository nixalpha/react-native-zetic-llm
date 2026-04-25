#include "HybridZeticAgentShadowTree.hpp"

#include <react/renderer/components/text/RawTextProps.h>
#include <react/renderer/components/textinput/BaseTextInputProps.h>
#include <react/renderer/components/view/ViewProps.h>
#include <react/renderer/core/LayoutableShadowNode.h>
#include <react/renderer/core/ShadowNode.h>
#include <react/renderer/uimanager/UIManagerBinding.h>

#include <algorithm>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

namespace margelo::nitro::zeticllm {
namespace {
using facebook::jsi::Runtime;
using facebook::jsi::Value;
using facebook::react::BaseTextInputProps;
using facebook::react::Float;
using facebook::react::LayoutMetrics;
using facebook::react::LayoutableShadowNode;
using facebook::react::Point;
using facebook::react::RawTextProps;
using facebook::react::RootShadowNode;
using facebook::react::ShadowNode;
using facebook::react::UIManagerBinding;
using facebook::react::ViewProps;

std::string escape(const std::string& value) {
  std::string result;
  result.reserve(std::min<size_t>(value.size(), 240));
  for (char c : value) {
    if (result.size() >= 240) {
      break;
    }
    switch (c) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
      case '\r':
      case '\t':
        result += ' ';
        break;
      default:
        result += c;
    }
  }
  return result;
}

LayoutMetrics layoutMetricsForNode(const ShadowNode& node) {
  if (auto layoutable = dynamic_cast<const LayoutableShadowNode*>(&node)) {
    return layoutable->getLayoutMetrics();
  }
  return facebook::react::EmptyLayoutMetrics;
}

void appendNode(
    std::ostringstream& stream,
    const std::shared_ptr<const ShadowNode>& node,
    int depth,
    Float absoluteX,
    Float absoluteY) {
  const auto metrics = layoutMetricsForNode(*node);
  const auto frame = metrics.frame;
  const auto x = absoluteX + frame.origin.x;
  const auto y = absoluteY + frame.origin.y;
  const auto indent = std::string(static_cast<size_t>(depth) * 2, ' ');

  stream << indent << node->getTag() << " " << node->getComponentName()
         << " rect=(" << std::fixed << std::setprecision(1) << x << "," << y << ","
         << frame.size.width << "," << frame.size.height << ")";

  const auto& props = node->getProps();

  if (auto viewProps = std::dynamic_pointer_cast<const ViewProps>(props)) {
    stream << " accessible=" << (viewProps->accessible ? "true" : "false");
    if (!viewProps->accessibilityRole.empty()) {
      stream << " role=\"" << escape(viewProps->accessibilityRole) << "\"";
    }
    if (!viewProps->accessibilityLabel.empty()) {
      stream << " label=\"" << escape(viewProps->accessibilityLabel) << "\"";
    }
    if (!viewProps->testId.empty()) {
      stream << " testID=\"" << escape(viewProps->testId) << "\"";
    }
  }

  if (auto rawTextProps = std::dynamic_pointer_cast<const RawTextProps>(props)) {
    if (!rawTextProps->text.empty()) {
      stream << " text=\"" << escape(rawTextProps->text) << "\"";
    }
  }

  if (auto inputProps = std::dynamic_pointer_cast<const BaseTextInputProps>(props)) {
    stream << " editable=" << (inputProps->editable && !inputProps->readOnly ? "true" : "false");
    if (!inputProps->text.empty()) {
      stream << " value=\"" << escape(inputProps->text) << "\"";
    }
    if (!inputProps->placeholder.empty()) {
      stream << " placeholder=\"" << escape(inputProps->placeholder) << "\"";
    }
  }

  stream << "\n";

  for (const auto& child : node->getChildren()) {
    appendNode(stream, child, depth + 1, x, y);
  }
}

std::string serializeRoot(const RootShadowNode::Shared& rootShadowNode) {
  std::ostringstream stream;
  stream << "Fabric shadow tree surface=" << rootShadowNode->getSurfaceId() << "\n";
  appendNode(stream, rootShadowNode, 0, 0, 0);
  return stream.str();
}
} // namespace

HybridZeticAgentShadowTree::HybridZeticAgentShadowTree()
    : HybridObject(TAG), HybridZeticAgentShadowTreeSpec() {}

HybridZeticAgentShadowTree::~HybridZeticAgentShadowTree() {
  unregisterObserver();
}

std::string HybridZeticAgentShadowTree::getLatestSnapshot() {
  std::lock_guard lock(mutex_);
  if (!latestSnapshot_.empty()) {
    return latestSnapshot_;
  }
  if (!fallbackSnapshot_.empty()) {
    return fallbackSnapshot_;
  }
  return "Fabric shadow tree unavailable. Call installShadowTreeObserver() on the JS thread after Fabric has initialized.";
}

void HybridZeticAgentShadowTree::setFallbackSnapshot(const std::string& snapshot) {
  std::lock_guard lock(mutex_);
  fallbackSnapshot_ = snapshot;
}

void HybridZeticAgentShadowTree::clear() {
  std::lock_guard lock(mutex_);
  latestSnapshot_.clear();
  fallbackSnapshot_.clear();
}

void HybridZeticAgentShadowTree::dispose() {
  unregisterObserver();
}

void HybridZeticAgentShadowTree::loadHybridMethods() {
  HybridZeticAgentShadowTreeSpec::loadHybridMethods();
  registerHybrids(this, [](Prototype& prototype) {
    prototype.registerRawHybridMethod(
        "installShadowTreeObserver",
        0,
        &HybridZeticAgentShadowTree::installShadowTreeObserverRaw);
  });
}

Value HybridZeticAgentShadowTree::installShadowTreeObserverRaw(
    Runtime& runtime,
    const Value&,
    const Value*,
    size_t) {
  auto binding = UIManagerBinding::getBinding(runtime);
  if (binding == nullptr) {
    return Value(false);
  }

  std::lock_guard lock(mutex_);
  if (!observer_) {
    observer_ = std::make_unique<MountObserver>(*this);
  }
  if (!registered_) {
    uiManager_ = &binding->getUIManager();
    uiManager_->registerMountHook(*observer_);
    registered_ = true;
  }
  return Value(true);
}

void HybridZeticAgentShadowTree::captureRoot(const RootShadowNode::Shared& rootShadowNode) noexcept {
  try {
    auto snapshot = serializeRoot(rootShadowNode);
    std::lock_guard lock(mutex_);
    latestSnapshot_ = std::move(snapshot);
  } catch (...) {
    // Mount hooks must not throw into React Native internals.
  }
}

void HybridZeticAgentShadowTree::unregisterObserver() noexcept {
  std::lock_guard lock(mutex_);
  if (registered_ && uiManager_ != nullptr && observer_) {
    try {
      uiManager_->unregisterMountHook(*observer_);
    } catch (...) {
    }
  }
  registered_ = false;
  uiManager_ = nullptr;
}

HybridZeticAgentShadowTree::MountObserver::MountObserver(HybridZeticAgentShadowTree& owner)
    : owner_(owner) {}

void HybridZeticAgentShadowTree::MountObserver::shadowTreeDidMount(
    const RootShadowNode::Shared& rootShadowNode,
    facebook::react::HighResTimeStamp) noexcept {
  owner_.captureRoot(rootShadowNode);
}

void HybridZeticAgentShadowTree::MountObserver::shadowTreeDidUnmount(
    facebook::react::SurfaceId surfaceId,
    facebook::react::HighResTimeStamp) noexcept {
  std::lock_guard lock(owner_.mutex_);
  owner_.latestSnapshot_ = "Fabric shadow tree surface=" + std::to_string(surfaceId) + " unmounted.";
}

} // namespace margelo::nitro::zeticllm

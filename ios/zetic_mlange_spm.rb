require 'xcodeproj'

module ReactNativeZeticLLM
  ZETIC_MLANGE_PACKAGE_URL = 'https://github.com/zetic-ai/ZeticMLangeiOS.git'
  ZETIC_MLANGE_VERSION = '1.6.0'
  ZETIC_MLANGE_PRODUCT = 'ZeticMLange'
  POD_TARGET_NAME = 'NitroZeticLlm'

  def self.install_zetic_mlange_spm!(installer)
    changed_projects = {}

    pod_target = installer.pods_project.targets.find { |target| target.name == POD_TARGET_NAME }
    if pod_target
      package_reference = ensure_package_reference!(installer.pods_project)
      ensure_product_dependency!(pod_target, package_reference)
      changed_projects[installer.pods_project] = true
    else
      log("warning: #{POD_TARGET_NAME} pod target was not found; skipping pod target SPM linkage")
    end

    each_user_application_target(installer) do |target|
      package_reference = ensure_package_reference!(target.project)
      ensure_product_dependency!(target, package_reference)
      changed_projects[target.project] = true
    end

    changed_projects.each_key(&:save)
  end

  def self.each_user_application_target(installer)
    installer.aggregate_targets.each do |aggregate_target|
      next unless aggregate_target.respond_to?(:user_targets)

      aggregate_target.user_targets.each do |target|
        next unless target.respond_to?(:product_type)
        next unless target.product_type == 'com.apple.product-type.application'

        yield target
      end
    end
  end

  def self.ensure_package_reference!(project)
    package_reference = project.root_object.package_references.find do |reference|
      normalize_url(reference.repositoryURL) == normalize_url(ZETIC_MLANGE_PACKAGE_URL)
    end

    unless package_reference
      package_reference = project.new(Xcodeproj::Project::Object::XCRemoteSwiftPackageReference)
      package_reference.repositoryURL = ZETIC_MLANGE_PACKAGE_URL
      project.root_object.package_references << package_reference
    end

    package_reference.requirement = {
      'kind' => 'exactVersion',
      'version' => ZETIC_MLANGE_VERSION,
    }

    package_reference
  end

  def self.ensure_product_dependency!(target, package_reference)
    product_dependency = target.package_product_dependencies.find do |dependency|
      dependency.product_name == ZETIC_MLANGE_PRODUCT &&
        normalize_url(dependency.package&.repositoryURL) == normalize_url(ZETIC_MLANGE_PACKAGE_URL)
    end

    unless product_dependency
      product_dependency = target.project.new(Xcodeproj::Project::Object::XCSwiftPackageProductDependency)
      product_dependency.product_name = ZETIC_MLANGE_PRODUCT
      product_dependency.package = package_reference
      target.package_product_dependencies << product_dependency
    end

    frameworks_phase = target.frameworks_build_phase
    return if frameworks_phase.files.any? { |file| file.product_ref == product_dependency }

    build_file = target.project.new(Xcodeproj::Project::Object::PBXBuildFile)
    build_file.product_ref = product_dependency
    frameworks_phase.files << build_file
  end

  def self.normalize_url(url)
    url.to_s.sub(%r{\.git\z}, '')
  end

  def self.log(message)
    if defined?(Pod::UI)
      Pod::UI.puts("[react-native-zetic-llm] #{message}")
    else
      puts("[react-native-zetic-llm] #{message}")
    end
  end
end

def react_native_zetic_llm_post_install(installer)
  ReactNativeZeticLLM.install_zetic_mlange_spm!(installer)
end

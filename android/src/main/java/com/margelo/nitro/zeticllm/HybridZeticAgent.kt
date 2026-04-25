package com.margelo.nitro.zeticllm

import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import com.facebook.proguard.annotations.DoNotStrip
import com.facebook.react.bridge.ReactApplicationContext
import com.margelo.nitro.core.Promise
import com.zeticai.mlange.*
import com.zeticai.mlange.core.cache.ModelCacheHandlingPolicy
import com.zeticai.mlange.core.model.APType
import com.zeticai.mlange.core.model.ModelLoadingStatus
import com.zeticai.mlange.core.model.llm.*
import org.json.JSONObject
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@DoNotStrip
class HybridZeticAgent : HybridZeticAgentSpec() {
  private val lock = Any()
  private val mainHandler = Handler(Looper.getMainLooper())
  private var model: ZeticMLangeLLMModel? = null
  private var listener: ((AgentEvent) -> Unit)? = null
  private var state = STATE_IDLE
  private var task = ""
  private var instruction = ""
  private var memory = ""
  private var actionsTaken = 0
  private var stopRequested = false
  private var paused = false
  private var isRunning = false
  private var nodeRects = emptyMap<String, Rect>()

  override fun loadModel(
    config: NativeLoadModelConfig,
    onDownload: ((Double) -> Unit)?
  ): Promise<Unit> {
    return Promise.async {
      setState(STATE_LOADING_MODEL)
      val context = ZeticLLMContextHolder.requireContext()
      val version = config.version?.toInt()
      val initOption = makeInitOption(config.initOption)
      val cachePolicy = makeCachePolicy(config.cacheHandlingPolicy)
      val progress = makeProgressCallback(onDownload)
      val statusChanged: (ModelLoadingStatus) -> Unit = {
        emit("model_status", "Zetic model loading status: $it")
      }

      val loaded = config.explicitRuntime?.let { explicit ->
        ZeticMLangeLLMModel(
          context = context,
          personalKey = config.personalKey,
          name = config.name,
          version = version,
          target = makeTarget(explicit.target),
          quantType = makeQuantType(explicit.quantType),
          apType = makeAPType(explicit.apType),
          onProgress = progress,
          onStatusChanged = statusChanged,
          cacheHandlingPolicy = cachePolicy,
          initOption = initOption
        )
      } ?: ZeticMLangeLLMModel(
        context = context,
        personalKey = config.personalKey,
        name = config.name,
        version = version,
        modelMode = makeModelMode(config.modelMode),
        dataSetType = makeDataSetType(config.dataSetType),
        onProgress = progress,
        onStatusChanged = statusChanged,
        cacheHandlingPolicy = cachePolicy,
        initOption = initOption
      )

      synchronized(lock) {
        model?.deinit()
        model = loaded
      }
      setState(STATE_IDLE)
      emit("model_loaded", "Zetic agent model loaded.")
    }
  }

  override fun start(task: String, options: AgentOptions?): Promise<Unit> {
    return Promise.async {
      val activeModel = synchronized(lock) {
        if (isRunning) {
          throw IllegalStateException("AGENT_RUNNING: The agent is already running.")
        }
        model ?: throw IllegalStateException("MODEL_NOT_LOADED: Call loadModel() before start().")
      }

      synchronized(lock) {
        this.task = task
        actionsTaken = 0
        stopRequested = false
        paused = false
        isRunning = true
      }

      val maxActions = options?.maxActions?.toInt()?.coerceAtLeast(1) ?: 12
      val maxRuntimeMs = options?.maxRuntimeMs?.toLong()?.coerceAtLeast(1_000L) ?: 120_000L
      val deadline = SystemClock.uptimeMillis() + maxRuntimeMs

      try {
        emit("started", "Agent started: $task")
        while (!shouldStop() && actionsTaken < maxActions && SystemClock.uptimeMillis() < deadline) {
          waitWhilePaused()
          if (shouldStop()) break

          setState(STATE_OBSERVING)
          val screen = snapshotBlocking(ZeticLLMContextHolder.requireReactContext())
          emit("snapshot", "Captured UI snapshot.", snapshot = screen)

          setState(STATE_THINKING)
          val response = generatePlan(activeModel, buildPrompt(task, screen))
          emit("llm_output", "LLM produced an action plan.", actionJson = response)

          val action = parseAction(response)
          if (options?.requireConfirmation == true && action.optString("type") != "done") {
            setState(STATE_PAUSED)
            paused = true
            emit("confirmation_required", "Action requires user confirmation.", actionJson = action.toString())
            waitWhilePaused()
          }

          if (shouldStop()) break
          setState(STATE_ACTING)
          val outcome = executeAction(action)
          actionsTaken += if (outcome.consumedAction) 1 else 0
          memory = (memory + "\n- ${action.optString("type", "unknown")}: ${outcome.message}").takeLast(4_000)
          emit("action", outcome.message, actionJson = action.toString())

          if (outcome.done) {
            break
          }
          SystemClock.sleep(350)
        }
      } finally {
        synchronized(lock) {
          isRunning = false
        }
        setState(if (shouldStop()) STATE_STOPPED else STATE_IDLE)
        emit("stopped", "Agent stopped after $actionsTaken action(s).")
      }
    }
  }

  override fun pause() {
    paused = true
    setState(STATE_PAUSED)
    emit("paused", "Agent paused.")
  }

  override fun resume() {
    paused = false
    setState(if (isRunning) STATE_OBSERVING else STATE_IDLE)
    emit("resumed", "Agent resumed.")
  }

  override fun stop() {
    stopRequested = true
    paused = false
    setState(STATE_STOPPED)
    emit("stop_requested", "Stop requested.")
  }

  override fun sendInstruction(text: String) {
    instruction = text
    emit("instruction", "User instruction updated: $text")
  }

  override fun snapshot(): Promise<String> {
    return Promise.async {
      snapshotBlocking(ZeticLLMContextHolder.requireReactContext())
    }
  }

  override fun getState(): AgentStateSnapshot =
    AgentStateSnapshot(state, task, instruction, actionsTaken.toDouble(), memory)

  override fun onEvent(callback: (AgentEvent) -> Unit) {
    listener = callback
  }

  private fun generatePlan(activeModel: ZeticMLangeLLMModel, prompt: String): String {
    activeModel.cleanUp()
    activeModel.run(prompt)
    val output = StringBuilder()
    while (!shouldStop()) {
      val next = activeModel.waitForNextToken()
      if (next.generatedTokens == 0) break
      output.append(next.token)
    }
    activeModel.cleanUp()
    return output.toString()
  }

  private fun buildPrompt(task: String, snapshot: String): String =
    """
    You are an on-device React Native UI agent. Decide exactly one next action.
    Task: $task
    Latest user instruction: ${instruction.ifBlank { "(none)" }}
    Recent memory:
    ${memory.ifBlank { "(none)" }}

    UI snapshot:
    $snapshot

    Return strict JSON only, no Markdown:
    {"type":"tap","nodeId":"node id from snapshot"}
    {"type":"tap","x":123,"y":456}
    {"type":"wait","ms":500}
    {"type":"askUser","question":"short question"}
    {"type":"done","reason":"why the task is complete"}
    """.trimIndent()

  private fun parseAction(text: String): JSONObject {
    val trimmed = text.trim()
    val start = trimmed.indexOf('{')
    val end = trimmed.lastIndexOf('}')
    if (start < 0 || end <= start) {
      return JSONObject().put("type", "askUser").put("question", "The model did not return a JSON action.")
    }
    val parsed = JSONObject(trimmed.substring(start, end + 1))
    return parsed.optJSONObject("action") ?: parsed
  }

  private data class ActionOutcome(
    val message: String,
    val done: Boolean = false,
    val consumedAction: Boolean = true
  )

  private fun executeAction(action: JSONObject): ActionOutcome {
    return when (action.optString("type")) {
      "tap" -> {
        val point = resolveTapPoint(action)
          ?: return ActionOutcome("Tap target could not be resolved: $action")
        injectTap(point.first, point.second)
        ActionOutcome("Tapped at ${point.first}, ${point.second}.")
      }
      "wait" -> {
        val ms = action.optLong("ms", 500L).coerceIn(0L, 5_000L)
        SystemClock.sleep(ms)
        ActionOutcome("Waited ${ms}ms.")
      }
      "askUser" -> {
        pause()
        ActionOutcome(action.optString("question", "Agent needs user input."), consumedAction = false)
      }
      "done" -> ActionOutcome(action.optString("reason", "Task complete."), done = true, consumedAction = false)
      else -> ActionOutcome("Unsupported action: ${action.optString("type")}", consumedAction = false)
    }
  }

  private fun resolveTapPoint(action: JSONObject): Pair<Float, Float>? {
    if (action.has("x") && action.has("y")) {
      return action.optDouble("x").toFloat() to action.optDouble("y").toFloat()
    }
    val nodeId = action.optString("nodeId")
    val rect = nodeRects[nodeId] ?: return null
    return rect.exactCenterX() to rect.exactCenterY()
  }

  private fun injectTap(x: Float, y: Float) {
    runOnMainSync {
      val activity = ZeticLLMContextHolder.requireReactContext().currentActivity
        ?: throw IllegalStateException("NO_ACTIVITY: Cannot inject tap without a current Activity.")
      val downTime = SystemClock.uptimeMillis()
      val down = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, x, y, 0)
      val up = MotionEvent.obtain(downTime, downTime + 50, MotionEvent.ACTION_UP, x, y, 0)
      try {
        activity.dispatchTouchEvent(down)
        activity.dispatchTouchEvent(up)
      } finally {
        down.recycle()
        up.recycle()
      }
    }
  }

  private fun snapshotBlocking(context: ReactApplicationContext): String {
    var result = ""
    runOnMainSync {
      val activity = context.currentActivity
        ?: throw IllegalStateException("NO_ACTIVITY: Cannot snapshot without a current Activity.")
      val root = activity.window.decorView
      val rects = LinkedHashMap<String, Rect>()
      val builder = StringBuilder()
      builder.appendLine("Android view hierarchy")
      appendView(builder, root, "0", 0, rects)
      nodeRects = rects
      result = builder.toString()
    }
    return result
  }

  private fun appendView(
    builder: StringBuilder,
    view: View,
    id: String,
    depth: Int,
    rects: MutableMap<String, Rect>
  ) {
    if (view.visibility != View.VISIBLE) return
    val rect = Rect()
    view.getGlobalVisibleRect(rect)
    rects[id] = rect
    val indent = "  ".repeat(depth)
    val className = view.javaClass.simpleName
    val text = (view as? TextView)?.text?.toString()?.takeIf { it.isNotBlank() }
    val label = view.contentDescription?.toString()?.takeIf { it.isNotBlank() }
    val clickable = view.isClickable || view.isFocusable
    builder.append(indent)
      .append(id)
      .append(" ")
      .append(className)
      .append(" rect=(${rect.left},${rect.top},${rect.width()},${rect.height()})")
      .append(" enabled=${view.isEnabled}")
      .append(" clickable=$clickable")
    if (label != null) builder.append(" label=\"").append(label.escapeForSnapshot()).append("\"")
    if (text != null) builder.append(" text=\"").append(text.escapeForSnapshot()).append("\"")
    builder.appendLine()

    if (view is ViewGroup) {
      for (index in 0 until view.childCount) {
        appendView(builder, view.getChildAt(index), "$id.$index", depth + 1, rects)
      }
    }
  }

  private fun String.escapeForSnapshot(): String =
    replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").take(240)

  private fun runOnMainSync(block: () -> Unit) {
    if (Looper.myLooper() == Looper.getMainLooper()) {
      block()
      return
    }

    var error: Throwable? = null
    val latch = CountDownLatch(1)
    mainHandler.post {
      try {
        block()
      } catch (caught: Throwable) {
        error = caught
      } finally {
        latch.countDown()
      }
    }
    if (!latch.await(5, TimeUnit.SECONDS)) {
      throw IllegalStateException("MAIN_THREAD_TIMEOUT: Timed out waiting for the UI thread.")
    }
    error?.let { throw it }
  }

  private fun waitWhilePaused() {
    while (paused && !stopRequested) {
      SystemClock.sleep(100)
    }
  }

  private fun shouldStop(): Boolean = stopRequested

  private fun setState(next: String) {
    state = next
  }

  private fun emit(
    type: String,
    message: String,
    snapshot: String? = null,
    actionJson: String? = null,
    progress: Double? = null
  ) {
    Log.d(TAG, "$type: $message")
    listener?.let { callback ->
      runCatching {
        callback(AgentEvent(type, state, message, snapshot, actionJson, progress))
      }.onFailure {
        Log.w(TAG, "Agent event callback failed.", it)
      }
    }
  }

  private fun makeProgressCallback(onDownload: ((Double) -> Unit)?): ((Float) -> Unit)? =
    { value ->
      val progress = value.toDouble().coerceIn(0.0, 1.0)
      emit("model_progress", String.format(Locale.US, "Model download %.1f%%", progress * 100.0), progress = progress)
      onDownload?.invoke(progress)
    }

  private fun normalize(value: String?): String = value?.trim()?.uppercase(Locale.US) ?: ""

  private fun makeModelMode(value: String?): LLMModelMode =
    when (normalize(value)) {
      "RUN_SPEED" -> LLMModelMode.RUN_SPEED
      "RUN_ACCURACY" -> LLMModelMode.RUN_ACCURACY
      else -> LLMModelMode.RUN_AUTO
    }

  private fun makeDataSetType(value: String?): LLMDataSetType? =
    when (normalize(value)) {
      "" -> null
      "MMLU" -> LLMDataSetType.MMLU
      "TRUTHFULQA" -> LLMDataSetType.TRUTHFULQA
      "CNN_DAILYMAIL" -> LLMDataSetType.CNN_DAILYMAIL
      "GSM8K" -> LLMDataSetType.GSM8K
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported dataSetType: $value")
    }

  private fun makeCachePolicy(value: String?): ModelCacheHandlingPolicy =
    when (normalize(value)) {
      "KEEP_EXISTING" -> ModelCacheHandlingPolicy.KEEP_EXISTING
      else -> ModelCacheHandlingPolicy.REMOVE_OVERLAPPING
    }

  private fun makeKVPolicy(value: String?): LLMKVCacheCleanupPolicy =
    when (normalize(value)) {
      "DO_NOT_CLEAN_UP" -> LLMKVCacheCleanupPolicy.DO_NOT_CLEAN_UP
      else -> LLMKVCacheCleanupPolicy.CLEAN_UP_ON_FULL
    }

  private fun makeInitOption(option: NativeLLMInitOption?): LLMInitOption =
    LLMInitOption(
      kvCacheCleanupPolicy = makeKVPolicy(option?.kvCacheCleanupPolicy),
      nCtx = option?.nCtx?.toInt() ?: 2048
    )

  private fun makeTarget(value: String): LLMTarget =
    when (normalize(value)) {
      "LLAMA_CPP" -> LLMTarget.LLAMA_CPP
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported target: $value")
    }

  private fun makeQuantType(value: String): LLMQuantType =
    when (normalize(value)) {
      "GGUF_QUANT_ORG" -> LLMQuantType.GGUF_QUANT_ORG
      "GGUF_QUANT_F16" -> LLMQuantType.GGUF_QUANT_F16
      "GGUF_QUANT_BF16" -> LLMQuantType.GGUF_QUANT_BF16
      "GGUF_QUANT_Q8_0" -> LLMQuantType.GGUF_QUANT_Q8_0
      "GGUF_QUANT_Q6_K" -> LLMQuantType.GGUF_QUANT_Q6_K
      "GGUF_QUANT_Q4_K_M" -> LLMQuantType.GGUF_QUANT_Q4_K_M
      "GGUF_QUANT_Q3_K_M" -> LLMQuantType.GGUF_QUANT_Q3_K_M
      "GGUF_QUANT_Q2_K" -> LLMQuantType.GGUF_QUANT_Q2_K
      "GGUF_QUANT_NUM_TYPES" -> LLMQuantType.GGUF_QUANT_NUM_TYPES
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported quantType: $value")
    }

  private fun makeAPType(value: String?): APType =
    when (normalize(value)) {
      "", "CPU" -> APType.CPU
      "GPU" -> APType.GPU
      "NPU" -> APType.NPU
      else -> throw IllegalArgumentException("INVALID_OPTION: Unsupported apType: $value")
    }

  companion object {
    private const val TAG = "ReactNativeZeticAgent"
    private const val STATE_IDLE = "idle"
    private const val STATE_LOADING_MODEL = "loading_model"
    private const val STATE_OBSERVING = "observing"
    private const val STATE_THINKING = "thinking"
    private const val STATE_ACTING = "acting"
    private const val STATE_PAUSED = "paused"
    private const val STATE_STOPPED = "stopped"
  }
}

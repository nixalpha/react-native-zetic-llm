package com.margelo.nitro.zeticllm

import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

internal object MLangeDownloadSize {
  private const val CACHE_DIR_NAME = "mlange_cache"

  fun <T> withDownloadedBytes(
    context: Context,
    tag: String,
    pollIntervalMs: Long = 500,
    onBytes: (Long) -> Unit,
    block: () -> T
  ): T {
    val cacheDir = File(context.filesDir, CACHE_DIR_NAME)
    val baseline = safeSize(cacheDir)
    val active = AtomicBoolean(true)

    runCatching { onBytes(0L) }

    val poller = thread(
      start = true,
      isDaemon = true,
      name = "zetic-mlange-download-size"
    ) {
      var lastBytes = Long.MIN_VALUE
      while (active.get()) {
        val downloaded = (safeSize(cacheDir) - baseline).coerceAtLeast(0L)
        if (downloaded != lastBytes) {
          lastBytes = downloaded
          runCatching { onBytes(downloaded) }
            .onFailure { Log.w(tag, "download byte callback failed.", it) }
        }
        SystemClock.sleep(pollIntervalMs)
      }
    }

    return try {
      block()
    } finally {
      val finalBytes = (safeSize(cacheDir) - baseline).coerceAtLeast(0L)
      runCatching { onBytes(finalBytes) }
        .onFailure { Log.w(tag, "download byte callback failed.", it) }
      active.set(false)
      poller.interrupt()
      runCatching { poller.join(100) }
    }
  }

  fun formatBytes(bytes: Long): String {
    val mib = bytes.toDouble() / (1024.0 * 1024.0)
    val gib = mib / 1024.0
    return if (gib >= 1.0) {
      String.format(java.util.Locale.US, "%.2f GiB", gib)
    } else {
      String.format(java.util.Locale.US, "%.1f MiB", mib)
    }
  }

  private fun safeSize(dir: File): Long = runCatching {
    if (!dir.exists()) {
      0L
    } else {
      dir.walkTopDown()
        .filter { it.isFile }
        .sumOf { it.length() }
    }
  }.getOrDefault(0L)
}

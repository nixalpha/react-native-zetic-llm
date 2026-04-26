package com.margelo.nitro.zeticllm

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

object MLangeDownloadSize {
  suspend fun <T> withDownloadedBytes(
    context: Context,
    pollIntervalMs: Long = 500,
    onBytes: (Long) -> Unit,
    block: suspend () -> T
  ): T = coroutineScope {
    val cacheDir = File(context.filesDir, "mlange_cache")
    val baseline = safeSize(cacheDir)

    runCatching { onBytes(0L) }

    val poller = launch(Dispatchers.IO) {
      while (isActive) {
        val downloaded = (safeSize(cacheDir) - baseline).coerceAtLeast(0L)
        runCatching { onBytes(downloaded) }
        delay(pollIntervalMs)
      }
    }

    try {
      block()
    } finally {
      poller.cancel()
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

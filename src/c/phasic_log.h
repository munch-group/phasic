/**
 * Unified logging system for phasic C code
 *
 * Provides logging functionality that integrates with Python logging system
 * via callback mechanism. Thread-safe and minimal overhead when disabled.
 *
 * Usage:
 *     PTD_LOG_DEBUG("Processing vertex %d with rate %f", v_idx, rate);
 *     PTD_LOG_INFO("Cache hit for hash %s", hash_hex);
 *     PTD_LOG_WARNING("Parameter out of range: %d", param_idx);
 *     PTD_LOG_ERROR("Failed to allocate memory for %zu bytes", size);
 *
 * Author: PtDAlgorithms Team
 * Date: 2025-11-08
 */

#ifndef PTD_LOG_H
#define PTD_LOG_H

#include <stdarg.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Logging levels matching Python logging module
 */
typedef enum {
    PTD_LOG_DEBUG = 10,
    PTD_LOG_INFO = 20,
    PTD_LOG_WARNING = 30,
    PTD_LOG_ERROR = 40,
    PTD_LOG_CRITICAL = 50
} ptd_log_level_t;

/**
 * Callback function type for forwarding logs to Python
 *
 * @param level Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
 * @param message Formatted log message (null-terminated string)
 */
typedef void (*ptd_log_callback_t)(ptd_log_level_t level, const char *message);

/**
 * Set the logging callback function
 *
 * This should be called during module initialization to connect C logging
 * to the Python logging system.
 *
 * @param callback Function to call for each log message (or NULL to disable)
 */
void ptd_set_log_callback(ptd_log_callback_t callback);

/**
 * Set minimum logging level
 *
 * Messages below this level will be discarded without calling the callback.
 *
 * @param level Minimum level to log (default: PTD_LOG_WARNING)
 */
void ptd_set_log_level(ptd_log_level_t level);

/**
 * Get current minimum logging level
 *
 * @return Current minimum log level
 */
ptd_log_level_t ptd_get_log_level(void);

/**
 * Core logging function
 *
 * Formats a message and sends it to the callback if level is sufficient.
 * Thread-safe. Maximum message length: 1024 characters.
 *
 * @param level Log level
 * @param format Printf-style format string
 * @param ... Format arguments
 */
void ptd_log(ptd_log_level_t level, const char *format, ...);

/**
 * Convenience macros for each log level
 *
 * These provide cleaner syntax and compiler can optimize away disabled levels.
 */
#define PTD_LOG_DEBUG(...) ptd_log(PTD_LOG_DEBUG, __VA_ARGS__)
#define PTD_LOG_INFO(...) ptd_log(PTD_LOG_INFO, __VA_ARGS__)
#define PTD_LOG_WARNING(...) ptd_log(PTD_LOG_WARNING, __VA_ARGS__)
#define PTD_LOG_ERROR(...) ptd_log(PTD_LOG_ERROR, __VA_ARGS__)
#define PTD_LOG_CRITICAL(...) ptd_log(PTD_LOG_CRITICAL, __VA_ARGS__)

/**
 * Conditional logging macros (for debugging)
 *
 * These are completely compiled out in release builds (when NDEBUG is defined)
 */
#ifndef NDEBUG
#define PTD_LOG_DEBUG_IF(cond, ...) do { if (cond) ptd_log(PTD_LOG_DEBUG, __VA_ARGS__); } while(0)
#else
#define PTD_LOG_DEBUG_IF(cond, ...) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* PTD_LOG_H */

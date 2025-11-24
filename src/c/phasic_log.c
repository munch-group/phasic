/**
 * Implementation of unified logging system for phasic C code
 */

#include "phasic_log.h"
#include <string.h>

// Platform-specific threading headers
#ifdef _WIN32
    #include <windows.h>
    static CRITICAL_SECTION g_log_mutex;
    static int g_log_mutex_initialized = 0;

    static void init_mutex_if_needed(void) {
        if (!g_log_mutex_initialized) {
            InitializeCriticalSection(&g_log_mutex);
            g_log_mutex_initialized = 1;
        }
    }

    #define LOCK_MUTEX() init_mutex_if_needed(); EnterCriticalSection(&g_log_mutex)
    #define UNLOCK_MUTEX() LeaveCriticalSection(&g_log_mutex)
#else
    #include <pthread.h>
    static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;

    #define LOCK_MUTEX() pthread_mutex_lock(&g_log_mutex)
    #define UNLOCK_MUTEX() pthread_mutex_unlock(&g_log_mutex)
#endif

/* Maximum message length */
#define PTD_LOG_MAX_MESSAGE_LEN 1024

/* Global logging state */
static ptd_log_callback_t g_log_callback = NULL;
static ptd_log_level_t g_log_level = PTD_LOG_WARNING;

void ptd_set_log_callback(ptd_log_callback_t callback) {
    LOCK_MUTEX();
    g_log_callback = callback;
    UNLOCK_MUTEX();
}

void ptd_set_log_level(ptd_log_level_t level) {
    LOCK_MUTEX();
    g_log_level = level;
    UNLOCK_MUTEX();
}

ptd_log_level_t ptd_get_log_level(void) {
    ptd_log_level_t level;
    LOCK_MUTEX();
    level = g_log_level;
    UNLOCK_MUTEX();
    return level;
}

void ptd_log(ptd_log_level_t level, const char *format, ...) {
    /* Early exit if level too low (no lock needed for read-only check) */
    if (level < g_log_level) {
        return;
    }

    /* Lock for the rest of the operation */
    LOCK_MUTEX();

    /* Check again with lock held (level might have changed) */
    if (level < g_log_level || g_log_callback == NULL) {
        UNLOCK_MUTEX();
        return;
    }

    /* Format the message */
    char buffer[PTD_LOG_MAX_MESSAGE_LEN];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    /* Ensure null termination */
    buffer[PTD_LOG_MAX_MESSAGE_LEN - 1] = '\0';

    /* Call the callback */
    g_log_callback(level, buffer);

    UNLOCK_MUTEX();
}

/**
 * Implementation of unified logging system for phasic C code
 */

#include "phasic_log.h"
#include <string.h>
#include <pthread.h>

/* Maximum message length */
#define PTD_LOG_MAX_MESSAGE_LEN 1024

/* Global logging state */
static ptd_log_callback_t g_log_callback = NULL;
static ptd_log_level_t g_log_level = PTD_LOG_WARNING;
static pthread_mutex_t g_log_mutex = PTHREAD_MUTEX_INITIALIZER;

void ptd_set_log_callback(ptd_log_callback_t callback) {
    pthread_mutex_lock(&g_log_mutex);
    g_log_callback = callback;
    pthread_mutex_unlock(&g_log_mutex);
}

void ptd_set_log_level(ptd_log_level_t level) {
    pthread_mutex_lock(&g_log_mutex);
    g_log_level = level;
    pthread_mutex_unlock(&g_log_mutex);
}

ptd_log_level_t ptd_get_log_level(void) {
    ptd_log_level_t level;
    pthread_mutex_lock(&g_log_mutex);
    level = g_log_level;
    pthread_mutex_unlock(&g_log_mutex);
    return level;
}

void ptd_log(ptd_log_level_t level, const char *format, ...) {
    /* Early exit if level too low (no lock needed for read-only check) */
    if (level < g_log_level) {
        return;
    }

    /* Lock for the rest of the operation */
    pthread_mutex_lock(&g_log_mutex);

    /* Check again with lock held (level might have changed) */
    if (level < g_log_level || g_log_callback == NULL) {
        pthread_mutex_unlock(&g_log_mutex);
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

    pthread_mutex_unlock(&g_log_mutex);
}

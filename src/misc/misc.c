#include <time.h>
#include <string.h>
#include "ubon_cstuff_version.h"
#include "log.h"

double time_now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

const char *ubon_cstuff_get_version(void)
{
    const char *version = (const char*)UBON_CSTUFF_VERSION;

    return version;
}

static const char *level_strings[] = {
  "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};

static void file_log_callback(log_Event *ev)
{
    char buf[64];
    buf[strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", ev->time)] = '\0';
    const char *file_str=ev->file;
    const char *file_str2=strstr(file_str,"/src/");

    if (file_str2!=0) file_str=file_str2+5;
    fprintf(
        (FILE*)ev->udata, "%s %-5s %s:%d: ",
        buf, level_strings[ev->level], file_str, ev->line);
    vfprintf((FILE*)ev->udata, ev->fmt, ev->ap);
    fprintf((FILE*)ev->udata, "\n");
    fflush((FILE*)ev->udata);
}

bool file_trace_enabled=false;
static FILE *file_trace_file=0;

void file_trace_enable(const char *filename)
{
    if (file_trace_enabled) return;
    file_trace_enabled=true;
    file_trace_file=fopen(filename,"wb");
    log_add_callback(file_log_callback,  (void *)file_trace_file, LOG_TRACE);
}

void file_trace(const char *txt)
{
    if (!file_trace_enabled) return;
    fprintf(file_trace_file, "%s\n", txt);
    fflush(file_trace_file);
}
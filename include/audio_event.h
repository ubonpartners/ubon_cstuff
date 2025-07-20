#ifndef __AUDIO_EVENT_H
#define __AUDIO_EVENT_H

#include "embedding.h"
#include "audioframe.h"

typedef struct {
    int class_index;
    float prob;
} audio_event_detection_t;

typedef struct audio_event audio_event_t;
audio_event_t *audio_event_create();
void audio_event_destroy(audio_event_t *ae);
embedding_t *audio_event_process(audio_event_t *ae, audioframe_t *fr);
int audio_event_postprocess(audio_event_t *ae, embedding_t *e, audio_event_detection_t *topk, int k);
const char *audio_event_class_name(audio_event_t *ae, int class_index);

#endif
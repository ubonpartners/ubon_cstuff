#ifndef __AUDIO_EVENT_H
#define __AUDIO_EVENT_H

#include "embedding.h"
#include "audioframe.h"

typedef enum audio_event_model
{
    AUDIO_EVENT_MODEL_EFFICIENTAT,
    AUDIO_EVENT_MODEL_TINYCLAP
} audio_event_model_t;

typedef struct {
    int class_index;
    float prob;
    embedding_t *emb;
} audio_event_detection_t;

typedef struct audio_event audio_event_t;
audio_event_t *audio_event_create(audio_event_model_t model);
void audio_event_destroy(audio_event_t *ae);
embedding_t *audio_event_process(audio_event_t *ae, audioframe_t *fr);
int audio_event_postprocess(audio_event_t *ae, embedding_t *e, audio_event_detection_t *topk, int k);
void audio_event_postprocess_tinyclap(audio_event_t *ae, embedding_t *e);
const char *audio_event_class_name(audio_event_t *ae, int class_index);

#endif
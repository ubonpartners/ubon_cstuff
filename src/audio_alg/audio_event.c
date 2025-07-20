#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include <assert.h>
#include "audio_event.h"
#include "infer_thread.h"
#include "efficientat_preprocess.h"
#include "efficientat_class_names.h"

struct audio_event
{
    infer_thread_t *inf_thr;
    EffatCPUHandle* efficientat_preprocess;
};

audio_event_t *audio_event_create()
{
    audio_event_t *ae=(audio_event_t *)malloc(sizeof(audio_event_t));
    memset(ae, 0, sizeof(audio_event_t));
    ae->inf_thr=infer_thread_start("/mldata/efficientat/trt/efficientat_m10_as.trt", 0, INFER_THREAD_AUX_TENSOR);
    PreprocConfig cfg = {32000,128,1024,800,320,0.0f,14000.0f,0.97f,1e-5f};
    ae->efficientat_preprocess = effat_cpu_create(&cfg);
    return ae;
}

void audio_event_destroy(audio_event_t *ae)
{
    if (!ae) return;
    effat_cpu_destroy(ae->efficientat_preprocess);
    infer_thread_destroy(ae->inf_thr);
    free(ae);
}

embedding_t *audio_event_process(audio_event_t *ae, audioframe_t *fr)
{
    assert(audioframe_get_sample_rate(fr)==32000);
    assert(audioframe_get_num_samples(fr)==32000); // 1 second
    image_t *mel_tensor=effat_cpu_preprocess(ae->efficientat_preprocess,
        audioframe_get_data(fr), audioframe_get_num_samples(fr));
    embedding_t *e=infer_thread_infer_embedding(ae->inf_thr, mel_tensor, 0);
    destroy_image(mel_tensor);
    return e;
}

static inline float sigmoidf(float x) {
    // Clamp to avoid expf overflow if extreme values occur
    if (x > 50.0f)  return 1.0f;
    if (x < -50.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

const char *audio_event_class_name(audio_event_t *ae, int index)
{
    if (index==-1) return "????";
    assert(index>=0 && index<NUM_EFFAT_CLASSES);
    return kClassNames527[index];
}

int audio_event_postprocess(audio_event_t *ae, embedding_t *e, audio_event_detection_t *topk, int k)
{
    float *logits=embedding_get_data(e);

    for (int i= 0; i < k; ++i) {
        topk[i].class_index= -1;
        topk[i].prob = -1.0f;
    }

    for(int i=0;i<NUM_EFFAT_CLASSES;i++)
    {
        float p=sigmoidf(logits[i]);
        if (p<topk[k-1].prob) continue;

        int pos=k;
        for (int j = k - 1; j >= 0; --j) {
            if (p > topk[j].prob) pos = j;
        }
        if (pos<k) memmove(topk+pos+1, topk+pos, (k-pos-1)*sizeof(audio_event_detection_t));
        topk[pos].class_index=i;
        topk[pos].prob=p;
    }
    return k;
}

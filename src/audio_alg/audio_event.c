#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include <assert.h>
#include "audio_event.h"
#include "infer_thread.h"
#include "preprocess.h"
#include "maths_stuff.h"
#include "efficientat_class_names.h"

struct audio_event
{
    audio_event_model_t model;
    int sample_rate;
    infer_thread_t *inf_thr;
    audio_preproc_t* pp;
};

audio_event_t *audio_event_create(audio_event_model_t model)
{
    audio_event_t *ae=(audio_event_t *)malloc(sizeof(audio_event_t));
    memset(ae, 0, sizeof(audio_event_t));
    ae->model=model;
    PreprocConfig cfg;
    if (model==AUDIO_EVENT_MODEL_EFFICIENTAT)
    {
        preproc_fill_efficientat_defaults(&cfg);
        ae->inf_thr=infer_thread_start("/mldata/efficientat/trt/efficientat_m10_as.trt", 0, INFER_THREAD_AUX_TENSOR);
        ae->sample_rate=32000;
    }
    else
    {
        preproc_fill_tinyclap_defaults(&cfg);
        ae->inf_thr=infer_thread_start("/mldata/tinyclap/trt/tinyclap_mel2emb.trt", 0, INFER_THREAD_AUX_TENSOR);
        ae->sample_rate=44100;
    }
    ae->pp = audio_preproc_create(&cfg);
    return ae;
}

void audio_event_destroy(audio_event_t *ae)
{
    if (!ae) return;
    audio_preproc_destroy(ae->pp);
    infer_thread_destroy(ae->inf_thr);
    free(ae);
}

embedding_t *audio_event_process(audio_event_t *ae, audioframe_t *fr)
{
    assert(audioframe_get_sample_rate(fr)==ae->sample_rate);
    image_t *mel_tensor=audio_preprocess(ae->pp, audioframe_get_data(fr), audioframe_get_num_samples(fr));
    embedding_t *e=infer_thread_infer_embedding(ae->inf_thr, mel_tensor, 0);
    image_destroy(mel_tensor);
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

typedef struct tinyclap_prompt
{
    embedding_t *emb;
    const char *txt;
} tinyclap_prompt_t;

static tinyclap_prompt_t *tc_prompts[10];
static int num_tc_prompts=0;

void audio_event_postprocess_tinyclap(audio_event_t *ae, embedding_t *e)
{
    static bool prompts_loaded=false;
    if (prompts_loaded==false)
    {
        prompts_loaded=true;
        FILE *f=fopen("/mldata/tinyclap/prompts/prompts.bin","rb");
        assert(f!=0);
        int sz=128+1024*4;
        uint8_t temp[sz];
        while(1)
        {
            if (fread(temp, 1, sz, f)<sz) break;
            printf("read prompt\n");
            tinyclap_prompt_t *p=(tinyclap_prompt_t *)malloc(sizeof(tinyclap_prompt_t));
            p->txt=strdup((char *)temp);
            p->emb=embedding_create(1024, 0);
            printf("here\n");
            embedding_set_data(p->emb, (float *)(temp+128), 1024, true);
            printf("Prompt %d : %s\n",num_tc_prompts,p->txt);
            tc_prompts[num_tc_prompts++]=p;
        }
    }

    float *p=embedding_get_data(e);
    vec_l2_norm_inplace(p, 1024);
    for(int i=0;i<num_tc_prompts;i++)
    {
        printf("%d) %30s %f\n",i,tc_prompts[i]->txt, vec_dot(p, embedding_get_data(tc_prompts[i]->emb), 1024));
    }
}

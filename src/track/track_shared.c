#include <stdint.h>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <unistd.h>
#include "track.h"
#include "image.h"
#include "motion_track.h"
#include "log.h"
#include "infer_thread.h"
#include "ctpl_stl.h"
#include "cuda_stuff.h"
#include "BYTETracker.h"
#include "utrack.h"
#include "simple_decoder.h"
#include "jpeg_thread.h"
#include "yaml_stuff.h"
#include "track_aux.h"
#include "track_shared.h"
#include "memory_stuff.h"

#define debugf if (0) log_debug

static void ctpl_thread_init(int id, pthread_barrier_t *barrier)
{
    debugf("starting track_shared ctpl thread %d",id);
    cuda_thread_init();
    pthread_barrier_wait(barrier);
}

track_shared_state_t *track_shared_state_create(const char *yaml_config)
{
    YAML::Node yaml_base=yaml_load(yaml_config);
    track_shared_state_t *tss=(track_shared_state_t *)malloc(sizeof(track_shared_state_t));
    assert(tss!=0);
    memset(tss, 0, sizeof(track_shared_state_t));
    debugf("Track shared state create");
    tss->config_yaml=yaml_to_cstring(yaml_base);
    tss->max_width=yaml_get_int_value(yaml_base["max_width"], 1280);
    tss->max_height=yaml_get_int_value(yaml_base["max_height"], 1280);
    tss->num_worker_threads=yaml_get_int_value(yaml_base["num_worker_threads"], 4);
    tss->motiontrack_min_roi_after_skip=yaml_get_float_value(yaml_base["motiontrack_min_roi_after_skip"], 0.01);
    tss->motiontrack_min_roi_after_nonskip=yaml_get_float_value(yaml_base["motiontrack_min_roi_after_nonskip"], 0.05);

    // create worker threads
    tss->thread_pool=new ctpl::thread_pool(tss->num_worker_threads);
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, tss->num_worker_threads+1);
    for(int i=0;i<tss->num_worker_threads;i++) tss->thread_pool->push(ctpl_thread_init, &barrier);
    pthread_barrier_wait(&barrier);
    pthread_barrier_destroy(&barrier);

    // set up aux inference from config
    YAML::Node auxInferenceConfigNode = yaml_base["inference_config"];
    if (auxInferenceConfigNode && auxInferenceConfigNode.IsDefined())
    {
        assert(auxInferenceConfigNode.IsMap());
        for (const auto& kv : auxInferenceConfigNode) {
            std::string aux_name = kv.first.as<std::string>();
            const YAML::Node& entry = kv.second;

            bool enabled=yaml_get_bool_value(entry["enabled"], true);
            if (!enabled) continue;

            infer_thread_type_t index=INFER_THREAD_DETECTION;
            if (strcmp(aux_name.c_str(), "detection")==0)
                index=INFER_THREAD_DETECTION;
            else if (strcmp(aux_name.c_str(), "face")==0)
                index=INFER_THREAD_AUX_FACE;
            else if (strcmp(aux_name.c_str(), "clip")==0)
                index=INFER_THREAD_AUX_CLIP;
             else if (strcmp(aux_name.c_str(), "fiqa")==0)
                index=INFER_THREAD_AUX_FIQA;
            else{
                log_fatal("unknown aux inference type %s",aux_name.c_str());
                assert(0);
            }

            const char *inference_yaml=yaml_to_cstring(entry);

            std::string trt_file=entry["trt"].as<std::string>();
            log_debug("create inference Name '%s' trt '%s'",aux_name.c_str(), trt_file.c_str());
            if (access(trt_file.c_str(), F_OK) != 0)
            {
                if (entry["trt_fallback"])
                {
                    trt_file=entry["trt_fallback"].as<std::string>();
                    log_warn("Could not find TRT file; trying fallback '%s'", trt_file.c_str());
                }
            }
            if (access(trt_file.c_str(), F_OK) != 0)
            {
                log_fatal("Could not find TRT file %s", trt_file.c_str());
                assert(0);
            }
            assert(tss->infer_thread[index]==0);
            tss->infer_thread[index]=infer_thread_start(trt_file.c_str(), inference_yaml, index);
            free((void*)inference_yaml);

            // type specific config
            switch(index)
            {
                case INFER_THREAD_DETECTION:
                {
                    infer_config_t config={};
                    config.det_thr=yaml_get_float_value(yaml_base["conf_thr"], 0.05);
                    config.set_det_thr=true;
                    config.nms_thr=yaml_get_float_value(yaml_base["nms_thr"], 0.45);
                    config.set_nms_thr=true;
                    infer_thread_configure(tss->infer_thread[INFER_THREAD_DETECTION], &config);
                    tss->md=infer_thread_get_model_description(tss->infer_thread[INFER_THREAD_DETECTION]);
                    break;
                }
                default:
                    ;
            }
        }
    }

    tss->jpeg_thread=jpeg_thread_create(tss->config_yaml);
    // done
    return tss;
}

model_description_t *track_shared_state_get_model_description(track_shared_state_t *tss)
{
    return tss->md;
}

void track_shared_state_destroy(track_shared_state_t *tss)
{
    if (!tss) return;
    tss->thread_pool->stop();
    delete tss->thread_pool;
    for(int i=0;i<INFER_THREAD_NUM_TYPES;i++) if (tss->infer_thread[i]) infer_thread_destroy(tss->infer_thread[i]);
    jpeg_thread_destroy(tss->jpeg_thread);
    free((void *)tss->config_yaml);
    free(tss);
}

const char *track_shared_state_get_stats(track_shared_state_t *tss)
{
    YAML::Node root;

    YAML::Node infer_threads;
    for(int i=0;i<INFER_THREAD_NUM_TYPES;i++)
    {
        if (tss->infer_thread[i])
        {
            YAML::Node node=infer_thread_stats_node(tss->infer_thread[i]);
            infer_threads[infer_thread_type_names[i]] = node;
        }
    }
    root["infer_threads"]=infer_threads;

    return yaml_to_cstring(root);
}

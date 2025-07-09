#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "memory_stuff.h"
#include "detections.h"
#include "match.h"
#include "log.h"
#include "misc.h"
#include "image.h"
#include "infer_thread.h"
#include "track.h"
#include "jpeg_thread.h"
#include "track_shared.h"
#include "track_aux.h"
#include "yaml_stuff.h"

#define debugf if (0) log_info

static std::once_flag initFlag;
static block_allocator_t *aux_data_allocator;

typedef struct aux_data
{
    uint64_t track_id;
    double last_live_time;
    embedding_t *face_embedding;
    jpeg_t *face_jpeg;
} aux_data_t;

using MapType = std::unordered_map<uint64_t, aux_data_t*>;

struct track_aux
{
    track_shared_state *tss;
    MapType* map;
    infer_thread_t *face_infer_thread;

    double main_jpeg_last_time;
    bool main_jpeg_enabled;
    int main_jpeg_max_width;
    int main_jpeg_max_height;
    int main_jpeg_max_quality;
    float main_jpeg_min_interval_seconds;

    bool face_embeddings_enabled;
    bool face_jpegs_enabled;
    float face_embeddings_min_quality;
    float face_min_quality_increment;
    float face_min_quality_multiple;
    int face_jpeg_min_width;
    int face_jpeg_min_height;
    int face_jpeg_max_width;
    int face_jpeg_max_height;
    int face_jpeg_quality;
};

static void track_aux_init()
{
    aux_data_allocator=block_allocator_create("aux_data", sizeof(aux_data_t));
}

track_aux_t *track_aux_create(track_shared_state *tss)
{
    std::call_once(initFlag, track_aux_init);

    track_aux_t *ta=(track_aux_t *)malloc(sizeof(track_aux_t));
    memset(ta, 0, sizeof(track_aux_t));
    ta->tss=tss;

    ta->map = new MapType;
    ta->face_infer_thread=tss->infer_thread[INFER_THREAD_AUX_FACE];

    YAML::Node yaml_base=yaml_load(tss->config_yaml);
    ta->main_jpeg_enabled=yaml_get_bool(yaml_base, false, 2, "main_jpeg", "enabled");
    ta->main_jpeg_max_width=yaml_get_int(yaml_base, 320, 2, "main_jpeg", "max_width");
    ta->main_jpeg_max_height=yaml_get_int(yaml_base, 320, 2, "main_jpeg", "max_height");
    ta->main_jpeg_max_quality=yaml_get_int(yaml_base, 90, 2, "main_jpeg", "quality");
    ta->main_jpeg_min_interval_seconds=yaml_get_float(yaml_base, 5.0, 2, "main_jpeg", "min_interval_seconds");
    ta->main_jpeg_last_time=-1000;

    ta->face_embeddings_enabled=yaml_get_bool(yaml_base, false, 2, "faces", "embeddings_enabled");
    ta->face_jpegs_enabled=yaml_get_bool(yaml_base, false, 2, "faces", "jpegs_enabled");
    ta->face_jpeg_min_width=yaml_get_int(yaml_base, 32, 2, "faces", "jpeg_min_width");
    ta->face_jpeg_min_height=yaml_get_int(yaml_base, 32, 2, "faces", "jpeg_min_height");
    ta->face_jpeg_max_width=yaml_get_int(yaml_base, 160, 2, "faces", "jpeg_max_width");
    ta->face_jpeg_max_height=yaml_get_int(yaml_base, 160, 2, "faces", "jpeg_max_height");
    ta->face_jpeg_quality=yaml_get_int(yaml_base, 90, 2, "faces", "jpeg_quality");
    ta->face_embeddings_min_quality=yaml_get_float(yaml_base, 0.01, 2, "faces", "min_quality");
    ta->face_min_quality_increment=yaml_get_float(yaml_base, 0.02, 2, "faces", "min_quality_increment");
    ta->face_min_quality_multiple=yaml_get_float(yaml_base, 1.2, 2, "faces", "min_quality_multiple");


    return ta;
}

void aux_data_destroy(aux_data_t *aux)
{
    if (aux->face_embedding) embedding_destroy(aux->face_embedding);
    if (aux->face_jpeg) jpeg_destroy(aux->face_jpeg);
    aux->face_embedding=0;
    aux->face_jpeg=0;
    block_free(aux);
}

void track_aux_destroy(track_aux_t *ta)
{
    for (auto& kv : *ta->map) {
        aux_data_destroy((aux_data_t *)kv.second);
    }
    delete ta->map;
    free(ta);
}

static aux_data_t *lookup_id(track_aux_t *ta, uint64_t id)
{
    auto it = ta->map->find(id);
    if (it != ta->map->end()) {
        // Found, return existing pointer
        return it->second;
    }
    return 0;
}

void track_aux_enable_face_embeddings(track_aux_t *ta, bool enabled, float min_quality)
{
    ta->face_embeddings_enabled=enabled;
    ta->face_embeddings_min_quality=min_quality;
}

void track_aux_run(track_aux_t *ta, image_t *img, detection_list_t *dets)
{
    if (dets==0) return;

    double time=dets->time;
    debugf("aux run %f",time);

    track_shared_state *tss=ta->tss;
    if (ta->main_jpeg_enabled && img!=0)
    {
        double time_delta=img->time-ta->main_jpeg_last_time;
        if (time_delta>ta->main_jpeg_min_interval_seconds)
        {
            dets->frame_jpeg=jpeg_thread_encode(tss->jpeg_thread, img, ROI_ONE, ta->main_jpeg_max_width, ta->main_jpeg_max_height);
            ta->main_jpeg_last_time=img->time;
        }
    }

    for(int i=0;i<dets->num_detections;i++)
    {
        detection_t *det=dets->det[i];
        aux_data_t *aux=lookup_id(ta, det->track_id);
        if (aux==0)
        {
            debugf("new id %lx; %d facepts", det->track_id,det->num_face_points);
            aux=(aux_data_t *)block_alloc(aux_data_allocator, sizeof(aux_data_t));
            memset(aux, 0, sizeof(aux_data_t));
            (*ta->map)[det->track_id]=aux;
            aux->track_id=det->track_id;
        }

        if (det->last_seen_time!=dets->time)
        {
            // this is not a new detection
            continue;
        }

        if ((ta->face_infer_thread!=0) && (ta->face_embeddings_enabled) && (det->num_face_points>0))
        {
            assert(det->num_face_points==5);
            float q=detection_face_quality_score(det);
            if (aux->face_embedding!=0 && embedding_is_ready(aux->face_embedding)==false)
                ; // wait for existing embedding
            else
            {
                float existing_score=embedding_get_quality(aux->face_embedding);
                float min_face_quality=ta->face_embeddings_min_quality;
                float min_bar=0;
                float w=det->subbox_x1-det->subbox_x0;
                float h=det->subbox_y1-det->subbox_y0;
                int iw=(int)(w*img->width);
                int ih=(int)(h*img->height);
                float expand=0.2f;
                if (iw>=ta->face_jpeg_min_width && ih>=ta->face_jpeg_min_height)
                {
                    if (existing_score>0)
                        min_bar=std::max(ta->face_min_quality_multiple*existing_score, existing_score+ta->face_min_quality_increment);
                    if (q>std::max(min_face_quality, min_bar))
                    {
                        if (aux->face_embedding) embedding_destroy(aux->face_embedding);
                        aux->face_embedding=infer_thread_infer_embedding(ta->face_infer_thread, img, det->face_points, det->num_face_points);
                        roi_t face_roi;
                        face_roi.box[0]=std::max(0.0f, det->subbox_x0-w*expand);
                        face_roi.box[1]=std::max(0.0f, det->subbox_y0-h*expand);
                        face_roi.box[2]=std::min(1.0f, det->subbox_x1+w*expand);
                        face_roi.box[3]=std::min(1.0f, det->subbox_y1+h*expand);
                        if (aux->face_jpeg) jpeg_destroy(aux->face_jpeg);
                        aux->face_jpeg=jpeg_thread_encode(tss->jpeg_thread, img, face_roi, ta->face_jpeg_max_width, ta->face_jpeg_max_height, ta->face_jpeg_quality);
                        embedding_set_quality(aux->face_embedding, q);
                        debugf("Generate new face embedding Q=%f->%f",existing_score,q);
                    }
                }
            }
        }

        if (aux->face_embedding!=0)
        {
            assert(det->face_embedding==0);
            det->face_embedding=embedding_reference(aux->face_embedding);
            assert(block_reference_count(aux->face_embedding)>=1);
        }
        if (aux->face_jpeg!=0)
        {
            assert(det->face_jpeg==0);
            det->face_jpeg=jpeg_reference(aux->face_jpeg);
        }
        aux->last_live_time=time;
    }

    for (auto it = ta->map->begin(); it != ta->map->end(); ) {
        // destroy all objects that are not being tracked any more
        // we can check if 'last_live_time' was updated above
        // - if not, it's no longer tracked
        if (it->second->last_live_time != time) {
            debugf("Destroy id %llx", (uint64_t)it->first);
            aux_data_destroy((aux_data_t *)it->second);
            it = ta->map->erase(it);  // erase returns the next valid iterator
        } else {
            ++it;
        }
    }
}
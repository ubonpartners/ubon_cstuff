#include <stdint.h>
#include <string.h>
#include "trackset.h"
#include "yaml_stuff.h"
#include "detections.h"

trackset_t *trackset_load(const char *path)
{
    trackset_t *ts=(trackset_t *)malloc(sizeof(trackset_t));
    memset(ts, 0, sizeof(trackset_t));

    YAML::Node root=YAML::Node( YAML::LoadFile(path) );

    auto m = root["metadata"];
    ts->metadata.classes        = m["classes"].as<std::vector<std::string>>();
    ts->metadata.frame_rate     = m["frame_rate"].as<double>();
    ts->metadata.width          = m["width"].as<int>();
    ts->metadata.height         = m["height"].as<int>();
    ts->metadata.original_video = m["original_video"].as<std::string>();

    // --- frames ---
    for (const auto& frameNode : root["frames"]) {
        // collect the timestamp
        double t = frameNode["frame_time"].as<double>();
        ts->frame_times.push_back(t);

        detection_list_t *detection_list=detection_list_create(frameNode["objects"].size());

        // collect all detections in this frame
        for (const auto& objPair : frameNode["objects"]) {
            // objPair.first is the object-ID string (e.g. "1", "10001"), if you need it:
            std::string id_str = objPair.first.as<std::string>();
            uint64_t id_num = std::stoull(id_str);

            const auto& info = objPair.second;
            const auto& box = info["box"];

            detection_t *det=detection_list_add_end(detection_list);
            det->x0=box[0].as<float>();
            det->y0=box[1].as<float>();
            det->x1=box[2].as<float>();
            det->y1=box[3].as<float>();
            det->track_id=id_num;

            det->cl = info["class"].as<int>();
            det->conf = info["conf"].as<float>();
        }
        ts->frame_detections.push_back(detection_list);
    }

    return ts;
}

void trackset_destroy(trackset_t *ts)
{
    for (std::vector<detection_list_t *>::iterator it = ts->frame_detections.begin(); it != ts->frame_detections.end(); ++it) {
        detection_list_destroy(*it);
    }
    ts->frame_detections.clear();
    ts->frame_times.clear();
    ts->metadata.classes.clear();
    ts->metadata.original_video.clear();
    free(ts);
}
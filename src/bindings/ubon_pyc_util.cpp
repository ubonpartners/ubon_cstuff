
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;  // <-- this line enables "_a" syntax

#include "image.h"
#include "infer.h"
#include "infer_thread.h"
#include "cuda_stuff.h"
#include "detections.h"
#include "log.h"
#include "misc.h"
#include "jpeg.h"
#include "kalman_tracker.h"
#include "fiqa.h"
#include "mota_metrics.h"

py::list convert_points(kp_t *pts, int n)
{
    py::list pt_list;
    for(int i=0;i<n;i++)
    {
        pt_list.append(pts[i].x);
        pt_list.append(pts[i].y);
        pt_list.append(pts[i].conf);
    }
    return pt_list;
}

py::list convert_float_array(float *a, int n)
{
    py::list pt_list;
    for(int i=0;i<n;i++) pt_list.append(a[i]);
    return pt_list;
}

py::dict convert_embedding(embedding_t *e)
{
    py::dict d;
    embedding_sync(e);
    d["data"]=convert_float_array(embedding_get_data(e), embedding_get_size(e));
    d["time"]=embedding_get_time(e);
    d["quality"]=embedding_get_quality(e);
    return d;
}

py::object convert_jpeg(jpeg_t *j)
{
    py::dict d;
    size_t sz=0;
    uint8_t *data=jpeg_get_data(j, &sz);
    if (sz==0) return py::none();
    d["data"]=py::bytes(reinterpret_cast<const char*>(data), sz);
    d["time"]=jpeg_get_time(j);
    d["quality"]=jpeg_get_quality(j);
    return d;
}

py::list convert_roi(roi_t roi)
{
    py::list box;
    box.append(roi.box[0]);
    box.append(roi.box[1]);
    box.append(roi.box[2]);
    box.append(roi.box[3]);
    return box;
}

py::dict convert_model_description(model_description_t* desc) {
    if (!desc) throw std::runtime_error("Failed to get model description");
    py::dict d;
    d["class_names"] = py::cast(desc->class_names);
    d["person_attribute_names"] = py::cast(desc->person_attribute_names);
    d["num_classes"] = desc->num_classes;
    d["num_person_attributes"] = desc->num_person_attributes;
    d["num_keypoints"] = desc->num_keypoints;
    d["max_batch"] = desc->max_batch;
    d["input_is_fp16"] = desc->input_is_fp16;
    d["output_is_fp16"] = desc->output_is_fp16;
    d["min_w"] = desc->min_w;
    d["max_w"] = desc->max_w;
    d["min_h"] = desc->min_h;
    d["max_h"] = desc->max_h;
    d["model_output_dims"] = py::make_tuple(desc->model_output_dims[0], desc->model_output_dims[1], desc->model_output_dims[2]);
    d["engineInfo"] = desc->engineInfo;
    return d;
}

py::dict convert_aux_model_description(aux_model_description_t* desc) {
    py::dict d;
    d["embedding_size"] = desc->embedding_size;
    d["max_batch"] = desc->max_batch;
    d["input_w"] = desc->input_w;
    d["input_h"] = desc->input_h;
    d["input_fp16"] = desc->input_fp16;
    d["output_fp16"] = desc->output_fp16;
    d["engineInfo"] = desc->engineInfo;
    return d;
}

py::object convert_detection_array(detection_t **dets, int num_dets)
{
    if (!dets)
        return py::none();

    py::list results;
    for (int j = 0; j < num_dets; ++j) {
        detection_t* det = dets[j];
        py::dict item;
        item["class"] = det->cl;
        item["confidence"] = det->conf;
        item["track_id"] = det->track_id;
        py::list box;
        box.append(det->x0);
        box.append(det->y0);
        box.append(det->x1);
        box.append(det->y1);
        item["box"] = box;
        if (det->subbox_conf>0)
        {
            py::list subbox;
            subbox.append(det->subbox_x0);
            subbox.append(det->subbox_y0);
            subbox.append(det->subbox_x1);
            subbox.append(det->subbox_y1);
            item["subbox"] = subbox;
            item["subbox_conf"] = det->subbox_conf;
        }
        assert(det->num_pose_points==0 || det->num_pose_points==17);
        assert(det->num_face_points==0 || det->num_face_points==5);
        if (det->num_face_points>0) item["face_points"]=convert_points(det->face_points, det->num_face_points);
        if (det->num_pose_points>0) item["pose_points"]=convert_points(det->pose_points, det->num_pose_points);
        if (det->num_attr>0) item["attrs"]=convert_float_array(det->attr, det->num_attr);
        if (det->reid_vector_len>0) item["reid_vector"] = convert_float_array(det->reid, det->reid_vector_len);
        if (det->face_embedding!=0) item["face_embedding"]= convert_embedding(det->face_embedding);
        if (det->clip_embedding!=0) item["clip_embedding"]= convert_embedding(det->clip_embedding);
        if (det->face_jpeg) item["face_jpeg"]=convert_jpeg(det->face_jpeg);
        if (det->clip_jpeg) item["clip_jpeg"]=convert_jpeg(det->clip_jpeg);
        if (det->fiqa_embedding) item["fiqa_score"]=fiqa_embedding_quality(det->fiqa_embedding);
        results.append(item);
    }
    return results;
}

py::object convert_detections(detection_list_t *dets)
{
    if (!dets)
        return py::none();  // <-- return Python None if dets is null
    return convert_detection_array(dets->det, dets->num_detections);
}

detection_t *parse_detection(const std::unordered_map<std::string, py::object>& py_det) {
    detection_t *det = detection_create();

    // Required: box
    auto box = py_det.at("box").cast<std::vector<float>>();
    if (box.size() != 4) {
        throw std::runtime_error("Expected 'box' to have 4 float elements (x0, y0, x1, y1).");
    }

    det->x0 = box[0];
    det->y0 = box[1];
    det->x1 = box[2];
    det->y1 = box[3];

    det->conf = py_det.at("confidence").cast<float>();
    det->cl = py_det.at("class").cast<unsigned short>();

    // Optional: track_id
    if (py_det.find("track_id") != py_det.end()) {
        det->track_id= py_det.at("track_id").cast<uint64_t>();
    }

    // Optional: face_points
    if (py_det.find("face_points") != py_det.end()) {
        auto face = py_det.at("face_points").cast<std::vector<float>>();
        if (face.size() != 5 * 3) {
            throw std::runtime_error("Expected 'face_points' to have 15 float elements (5 keypoints x 3).");
        }

        det->num_face_points = 5;
        for (int i = 0; i < 5; ++i) {
            det->face_points[i].x = face[i * 3 + 0];
            det->face_points[i].y = face[i * 3 + 1];
            det->face_points[i].conf = face[i * 3 + 2];
        }
    }

    // Optional: pose_points
    if (py_det.find("pose_points") != py_det.end()) {
        auto pose = py_det.at("pose_points").cast<std::vector<float>>();
        if (pose.size() != 17 * 3) {
            throw std::runtime_error("Expected 'pose_points' to have 51 float elements (17 keypoints x 3).");
        }

        det->num_pose_points = 17;
        for (int i = 0; i < 17; ++i) {
            det->pose_points[i].x = pose[i * 3 + 0];
            det->pose_points[i].y = pose[i * 3 + 1];
            det->pose_points[i].conf = pose[i * 3 + 2];
        }
    }

    return det;
};

py::dict convert_metric_results(metric_results_t *res) {
    py::dict d;
    d["num_frames"]          = res->num_frames;
    d["num_objects"]         = res->num_objects;
    d["mostly_tracked"]       = res->mostly_tracked;
    d["partially_tracked"]    = res->partially_tracked;
    d["mostly_lost"]          = res->mostly_lost;
    d["num_false_positives"]  = res->num_false_positives;
    d["num_misses"]           = res->num_misses;
    d["num_switches"]         = res->num_switches;
    d["num_fragmentations"]   = res->num_fragmentations;
    d["num_unique_objects"]   = res->num_unique_objects;
    d["num_matches"]          = res->num_matches;
    d["missed"]               = res->missed;
    d["fp_tracks"]            = res->fp_tracks;
    d["total_iou"]            = res->total_iou;
    d["recall"]               = res->recall;
    d["precision"]            = res->precision;
    d["mota"]                 = res->mota;
    d["motp"]                 = res->motp;
    d["idfp"]                 = res->idfp;
    d["idfn"]                 = res->idfn;
    d["idtp"]                 = res->idtp;
    d["idp"]                  = res->idp;
    d["idr"]                  = res->idr;
    d["idf1"]                 = res->idf1;
    return d;
}
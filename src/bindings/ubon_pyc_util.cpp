
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

py::object convert_detections(detection_list_t *dets)
{
    if (!dets)
        return py::none();  // <-- return Python None if dets is null

    py::list results;
    for (int j = 0; j < dets->num_detections; ++j) {
        detection_t* det = dets->det[j];
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
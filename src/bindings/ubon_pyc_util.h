#pragma once

py::list convert_points(kp_t *pts, int n);
py::list convert_float_array(float *a, int n);
py::dict convert_embedding(embedding_t *e);
py::object convert_jpeg(jpeg_t *j);
py::list convert_roi(roi_t roi);
py::dict convert_model_description(model_description_t* desc);
py::dict convert_aux_model_description(aux_model_description_t* desc);
py::object convert_detection_array(detection_t **dets, int num_dets);
py::object convert_detections(detection_list_t *dets);
py::dict convert_metric_results(metric_results_t *res);

detection_t *parse_detection(const std::unordered_map<std::string, py::object>& py_det);
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
#include "simple_decoder.h"
#include "display.h"
#include "nvof.h"
#include "log.h"
#include "misc.h"
#include "pcap_decoder.h"
#include "profile.h"
#include "infer_aux.h"
#include "jpeg.h"
#include "track.h"
#include "motion_track.h"

// to build: python setup.py build_ext --inplace

class c_image {
public:
    image_t* img;

    c_image(int width, int height, image_format_t fmt) {
        img = create_image(width, height, fmt);
    }

    ~c_image() {
        destroy_image(img);
    }

    c_image(image_t* ptr) {
        img = ptr;
    }

    std::pair<int, int> size() const {
        return {img->height, img->width};
    }

    uint64_t timestamp() const {
        return img->timestamp;
    }

    image_format_t format() const {
        return img->format;
    }

    std::string format_name() const {
        return image_format_name(img->format);
    }

    static std::shared_ptr<c_image> from_numpy(py::array_t<uint8_t> input_rgb) {
        py::buffer_info buf_info = input_rgb.request();
        auto* src_ptr = static_cast<uint8_t*>(buf_info.ptr);
        int height = buf_info.shape[0];
        int width = buf_info.shape[1];

        if (input_rgb.ndim() != 3 || input_rgb.shape(2) != 3)
            throw std::runtime_error("Input must be HxWx3 uint8 numpy array");
        if (!input_rgb.flags() & py::array::c_style)
            throw std::runtime_error("Input array must be C-contiguous");

        image_t* img = create_image(width, height, IMAGE_FORMAT_RGB24_HOST);
        uint8_t* dst_ptr = img->rgb;
        for (int y = 0; y < height; ++y) {
            uint8_t* row_dst = dst_ptr + y * img->stride_rgb;
            uint8_t* row_src = src_ptr + y * width * 3;
            std::memcpy(row_dst, row_src, width * 3);
        }
        image_t *image_device=image_convert(img, IMAGE_FORMAT_RGB24_DEVICE);
        destroy_image(img);

        return std::make_shared<c_image>(image_device);
    }

    std::shared_ptr<c_image> scale(int w, int h) {
        image_t* scaled = image_scale(img, w, h);
        return std::make_shared<c_image>(scaled);
    }

    std::shared_ptr<c_image> convert(image_format_t fmt) {
        image_t* converted = image_convert(img, fmt);
        return std::make_shared<c_image>(converted);
    }

    void display(const char *name) {
        display_image(name, img);
    }

    uint32_t hash() {
        return image_hash(img);
    }

    void sync() {
        image_sync(img);
    }

    std::shared_ptr<c_image> blur() {
        image_t* blurred = image_blur(img);
        return std::make_shared<c_image>(blurred);
    }

    std::shared_ptr<c_image> mad_4x4(std::shared_ptr<c_image> other) {
        image_t* img_other = other->raw();
        image_t* mad_img=image_mad_4x4(img, img_other);
        return std::make_shared<c_image>(mad_img);
    }

    std::shared_ptr<c_image> crop(int x, int y, int w, int h) {
        image_t* cropped = image_crop(img, x, y, w, h);
        return std::make_shared<c_image>(cropped);
    }

    std::shared_ptr<c_image> blend(std::shared_ptr<c_image> other,
                                    int sx, int sy, int w, int h,
                                    int dx, int dy) {
        image_t* img_other = other->raw();
        image_t* blended=image_blend(img, img_other, sx, sy, w, h, dx, dy);
        return std::make_shared<c_image>(blended);
    }

    py::array_t<uint8_t> to_numpy() {
        if ((img->format==IMAGE_FORMAT_MONO_DEVICE)
            ||(img->format==IMAGE_FORMAT_MONO_HOST))
        {
            image_t* tmp = image_convert(img, IMAGE_FORMAT_MONO_HOST);
            image_sync(tmp);
            auto result = py::array_t<uint8_t>({tmp->height, tmp->width});
            auto buf = result.mutable_unchecked<2>();

            for (int y = 0; y < tmp->height; ++y)
                for (int x = 0; x < tmp->width; ++x)
                    buf(y, x) = tmp->y[x + tmp->stride_y * y];

            destroy_image(tmp);
            return result;
        }
        else
        {
            image_t* tmp = image_convert(img, IMAGE_FORMAT_RGB24_HOST);
            image_sync(tmp);  // Ensure data is on host

            // Create output NumPy array with shape [H, W, 3]
            auto result = py::array_t<uint8_t>({tmp->height, tmp->width, 3});
            py::buffer_info buf_info = result.request();
            uint8_t* dst_ptr = static_cast<uint8_t*>(buf_info.ptr);

            for (int y = 0; y < tmp->height; ++y) {
                uint8_t* src_row = tmp->rgb + y * tmp->stride_rgb;
                uint8_t* dst_row = dst_ptr + y * tmp->width * 3;
                std::memcpy(dst_row, src_row, tmp->width * 3);
            }

            destroy_image(tmp);
            return result;
        }
    }

    image_t* raw() { return img; }
};

std::shared_ptr<c_image> c_load_jpeg(const char *file) {
        image_t *jpg=load_jpeg(file);
        return std::make_shared<c_image>(jpg);
    }

static void pop_float(py::dict& cfg, const char* key, float& dst, bool& flag) {
    if (cfg.contains(key)) {
        dst = cfg[key].cast<float>();
        flag = true;
        cfg.attr("pop")(key);
    }
};

static void pop_int(py::dict& cfg, const char* key, int& dst, bool& flag) {
    if (cfg.contains(key)) {
        dst = cfg[key].cast<int>();
        flag = true;
        cfg.attr("pop")(key);
    }
};

static void pop_bool(py::dict& cfg, const char* key, bool& dst, bool& flag) {
    if (cfg.contains(key)) {
        dst = cfg[key].cast<bool>();
        flag = true;
        cfg.attr("pop")(key);
    }
};

static py::list convert_points(kp_t *pts, int n)
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

static py::list convert_attributes(float *a, int n)
{
    py::list pt_list;
    for(int i=0;i<n;i++) pt_list.append(a[i]);
    return pt_list;
}

static void apply_infer_config(py::dict cfg_dict, infer_config_t& config) {
    py::dict cfg_copy(cfg_dict);  // Mutable copy

    pop_float(cfg_copy, "det_thr", config.det_thr, config.set_det_thr);
    pop_float(cfg_copy, "nms_thr", config.nms_thr, config.set_nms_thr);
    pop_bool(cfg_copy, "use_cuda_nms", config.use_cuda_nms, config.set_use_cuda_nms);
    pop_bool(cfg_copy, "fuse_face_person", config.fuse_face_person, config.set_fuse_face_person);
    pop_bool(cfg_copy, "allow_upscale", config.allow_upscale, config.set_allow_upscale);
    pop_int(cfg_copy, "limit_max_batch", config.limit_max_batch, config.set_limit_max_batch);
    pop_int(cfg_copy, "limit_max_width", config.limit_max_width, config.set_limit_max_width);
    pop_int(cfg_copy, "limit_max_height", config.limit_max_height, config.set_limit_max_height);
    pop_int(cfg_copy, "limit_min_width", config.limit_min_width, config.set_limit_min_width);
    pop_int(cfg_copy, "limit_min_height", config.limit_min_height, config.set_limit_min_height);
    pop_int(cfg_copy, "max_detections", config.max_detections, config.set_max_detections);

    // Check for unknown keys
    if (py::len(cfg_copy) > 0) {
        std::string unknown_keys;
        for (auto item : cfg_copy) {
            unknown_keys += py::str(item.first).cast<std::string>() + " ";
        }
        throw std::runtime_error("Unknown config keys: " + unknown_keys);
    }
}

static py::list convert_roi(roi_t roi)
{
    py::list box;
    box.append(roi.box[0]);
    box.append(roi.box[1]);
    box.append(roi.box[2]);
    box.append(roi.box[3]);
    return box;
}

static py::dict convert_model_description(model_description_t* desc) {
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

static py::object convert_detections(detection_list_t *dets)
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
        assert(det->num_pose_points==0 || det->num_pose_points==17);
        assert(det->num_face_points==0 || det->num_face_points==5);
        if (det->num_face_points>0) item["face_points"]=convert_points(det->face_points, det->num_face_points);
        if (det->num_pose_points>0) item["pose_points"]=convert_points(det->pose_points, det->num_pose_points);
        if (det->num_attr>0) item["attrs"]=convert_attributes(det->attr, det->num_attr);
        results.append(item);
    }
    return results;
}

class c_infer {
private:
    py::list infer_and_convert_batch(image_t** imgs, int num) {
        std::vector<detection_list_t*> dets(num, nullptr);
        infer_batch(inf, imgs, dets.data(), num);

        py::list all_results;
        for (int i = 0; i < num; ++i) {
            py::list results;
            if (dets[i]) {
                results=convert_detections(dets[i]);
                detection_list_destroy(dets[i]);
            }
            all_results.append(results);
        }
        //print_detection_stats();
        return all_results;
    }
public:
    infer_t* inf;

    c_infer(const std::string& trt_file, const std::string& yaml_file) {
        inf = infer_create(trt_file.c_str(), yaml_file.c_str());
        if (!inf) {
            throw std::runtime_error("Failed to create infer_t");
        }
    }

    ~c_infer() {
        if (inf) infer_destroy(inf);
    }

    py::list run(std::shared_ptr<c_image> image_obj) {
        image_t* img = image_obj->raw();
        image_t* img_arr[1] = { img };
        py::list result = infer_and_convert_batch(img_arr, 1);
        return result[0].cast<py::list>();  // return the first (and only) result
    }

    py::list run_batch(const std::vector<std::shared_ptr<c_image>>& images) {
        int num = static_cast<int>(images.size());
        std::vector<image_t*> img_ptrs(num);
        for (int i = 0; i < num; ++i) {
            img_ptrs[i] = images[i]->raw();
        }
        return infer_and_convert_batch(img_ptrs.data(), num);
    }

    void configure(py::dict cfg_dict) {
        infer_config_t config{};
        apply_infer_config(cfg_dict, config);
        infer_configure(inf, &config);
    }

    py::dict get_model_description() {
        return convert_model_description(infer_get_model_description(inf));
    }

    infer_t* raw() { return inf; }
};

class c_infer_thread {
public:
    infer_thread_t* thread;

    c_infer_thread(const std::string& trt_file, const std::string& yaml_file, py::dict config_dict) {
        infer_config_t config{};
        apply_infer_config(config_dict, config);

        thread = infer_thread_start(trt_file.c_str(), yaml_file.c_str(), &config);
        if (!thread) {
            throw std::runtime_error("Failed to start infer_thread");
        }
    }

    ~c_infer_thread() {
        if (thread) infer_thread_destroy(thread);
    }

    py::object infer_async(std::shared_ptr<c_image> image_obj, py::list roi_list) {
        roi_t roi;
        if (py::len(roi_list) != 4)
            throw std::runtime_error("ROI must have 4 elements: [x0, y0, x1, y1]");
        for (int i = 0; i < 4; ++i)
            roi.box[i] = roi_list[i].cast<float>();

        auto h = infer_thread_infer_async(thread, image_obj->raw(), roi);
        return py::capsule(h, "infer_thread_result_handle");
    }

    py::dict wait(py::object result_handle_capsule) {
        auto* h = static_cast<infer_thread_result_handle_t*>(result_handle_capsule.cast<py::capsule>().get_pointer());

        infer_thread_result_data_t result_data;
        infer_thread_wait_result(h, &result_data);

        py::dict result;
        py::list box;
        for (int i = 0; i < 4; ++i)
            box.append(result_data.inference_roi.box[i]);

        //result["queue_time"] = result_data.queue_time;
        //result["inference_time"] = result_data.inference_time;
        result["inference_roi"] = box;

        // Convert detections
        py::list detections;
        if (result_data.dets) {
            for (int j = 0; j < result_data.dets->num_detections; ++j) {
                detection_t* det = result_data.dets->det[j];
                py::dict item;
                item["class"] = det->cl;
                item["confidence"] = det->conf;
                py::list det_box;
                det_box.append(det->x0);
                det_box.append(det->y0);
                det_box.append(det->x1);
                det_box.append(det->y1);
                item["box"] = det_box;
                if (det->num_face_points > 0)
                    item["face_points"] = convert_points(det->face_points, det->num_face_points);
                if (det->num_pose_points > 0)
                    item["pose_points"] = convert_points(det->pose_points, det->num_pose_points);
                if (det->num_attr > 0)
                    item["attrs"] = convert_attributes(det->attr, det->num_attr);
                detections.append(item);
            }
            detection_list_destroy(result_data.dets);
        }
        result["detections"] = detections;
        return result;
    }

    py::dict get_stats() {
        infer_thread_stats_t stats{};
        infer_thread_get_stats(thread, &stats);

        py::dict d;
        d["total_batches"] = stats.total_batches;
        d["total_images"] = stats.total_images;
        d["total_roi_area"] = stats.total_roi_area;
        d["mean_batch_size"] = stats.mean_batch_size;
        d["mean_roi_area"] = stats.mean_roi_area;

        py::list hist, total_time, time_per_infer;
        for (int i = 0; i < INFER_THREAD_MAX_BATCH; ++i) {
            hist.append(stats.batch_size_histogram[i]);
            total_time.append(stats.batch_size_histogram_total_time[i]);
            time_per_infer.append(stats.batch_size_histogram_time_per_inference[i]);
        }
        d["batch_size_histogram"] = hist;
        d["batch_size_histogram_total_time"] = total_time;
        d["batch_size_histogram_time_per_inference"] = time_per_infer;
        return d;
    }

    py::dict get_model_description() {
        return convert_model_description(infer_thread_get_model_description(thread));
    }
};

class c_infer_aux {
public:
    infer_aux_t* aux;
    int embedding_size;

    c_infer_aux(const std::string& trt_file) {
        aux = infer_aux_create(trt_file.c_str());
        if (!aux) {
            throw std::runtime_error("Failed to create infer_aux_t");
        }

        aux_model_description_t* desc = infer_aux_get_model_description(aux);
        if (!desc) {
            throw std::runtime_error("Failed to get aux model description");
        }

        embedding_size = desc->embedding_size;
    }

    ~c_infer_aux() {
        if (aux) infer_aux_destroy(aux);
    }

    py::list run(std::shared_ptr<c_image> image_obj, py::object keypoints = py::none()) {
        image_t* img = image_obj->raw();
        image_t* imgs[1] = { img };

        float* kp_ptr = nullptr;
        std::vector<float> kp_buf;

        if (!keypoints.is_none()) {
            py::list kp_list = keypoints;
            kp_buf.resize(kp_list.size());
            for (size_t i = 0; i < kp_buf.size(); ++i)
                kp_buf[i] = kp_list[i].cast<float>();
            kp_ptr = kp_buf.data();
        }

        float* output = infer_aux_batch(aux, imgs, kp_ptr, 1);

        py::list result;
        for (int i = 0; i < embedding_size; ++i)
            result.append(output[i]);

        return result;
    }

    py::list run_batch(const std::vector<std::shared_ptr<c_image>>& images, py::object all_keypoints = py::none()) {
        int num = static_cast<int>(images.size());
        std::vector<image_t*> img_ptrs(num);
        for (int i = 0; i < num; ++i)
            img_ptrs[i] = images[i]->raw();

        float* kp_ptr = nullptr;
        std::vector<float> kp_data;

        if (!all_keypoints.is_none()) {
            py::list kp_list = all_keypoints;
            kp_data.resize(kp_list.size());
            for (size_t i = 0; i < kp_data.size(); ++i)
                kp_data[i] = kp_list[i].cast<float>();
            kp_ptr = kp_data.data();
        }

        float* output = infer_aux_batch(aux, img_ptrs.data(), kp_ptr, num);

        py::list results;
        for (int i = 0; i < num; ++i) {
            py::list embedding;
            for (int j = 0; j < embedding_size; ++j)
                embedding.append(output[i * embedding_size + j]);
            results.append(embedding);
        }

        return results;
    }

    py::dict get_model_description() {
        aux_model_description_t* desc = infer_aux_get_model_description(aux);
        if (!desc) throw std::runtime_error("Failed to get aux model description");

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

    infer_aux_t* raw() { return aux; }
};

static auto parse_detection = [](const std::unordered_map<std::string, py::object>& py_det) -> detection_t* {
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

std::pair<std::vector<int>, std::vector<int>> match_box_iou_wrapper(
    const std::vector<std::unordered_map<std::string, py::object>>& py_dets_a,
    const std::vector<std::unordered_map<std::string, py::object>>& py_dets_b,
    float iou_thr, match_type_t match_type)
{
    int num_a=py_dets_a.size();
    int num_b=py_dets_b.size();
    detection_t *dets_a[num_a], *dets_b[num_b];
    for(int i=0;i<num_a;i++) dets_a[i]=parse_detection(py_dets_a[i]);
    for(int i=0;i<num_b;i++) dets_b[i]=parse_detection(py_dets_b[i]);

    std::vector<uint16_t> out_a_idx(num_a);
    std::vector<uint16_t> out_b_idx(num_b);

    int N = match_box_iou(
        dets_a, num_a,
        dets_b, num_b,
        out_a_idx.data(), out_b_idx.data(),
        iou_thr, match_type
    );

    for(int i=0;i<num_a;i++) detection_destroy(dets_a[i]);
    for(int i=0;i<num_b;i++) detection_destroy(dets_b[i]);

    std::vector<int> out_a(out_a_idx.begin(), out_a_idx.begin() + N);
    std::vector<int> out_b(out_b_idx.begin(), out_b_idx.begin() + N);

    return {out_a, out_b};
}

class c_decoder {
    public:
        simple_decoder_t* dec;
        std::vector<std::shared_ptr<c_image>> current_output;
        uint64_t dec_time;
        uint64_t dec_time_increment;

        c_decoder(simple_decoder_codec_t codec) {
            // Create the decoder, register our static callback, pass `this` as context
            dec = simple_decoder_create(this, &c_decoder::on_frame_static, codec);
            dec_time=0;
            dec_time_increment=90000/30;
            if (!dec) {
                throw std::runtime_error("Failed to create decoder");
            }
        }

        ~c_decoder() {
            if (dec) simple_decoder_destroy(dec);
        }

        void set_framerate(double fps) {
            dec_time_increment=(uint64_t)(90000.0/fps);
        }

        py::list decode(py::bytes bitstream) {
            current_output.clear();

            char* buffer;
            ssize_t length;
            PyBytes_AsStringAndSize(bitstream.ptr(), &buffer, &length);

            simple_decoder_decode(dec, reinterpret_cast<uint8_t*>(buffer), static_cast<int>(length));

            py::list result;
            for (const auto& img : current_output) {
                result.append(img);
            }
            return result;
        }

    private:
        static void on_frame_static(void* context, image_t* decoded_frame) {
            auto* self = static_cast<c_decoder*>(context);
            self->on_frame(decoded_frame);
        }

        void on_frame(image_t* img) {
            // Reference the frame so we manage its lifetime cleanly
            img->timestamp=dec_time;
            dec_time+=dec_time_increment;
            current_output.emplace_back(std::make_shared<c_image>(image_reference(img)));
        }
    };

class c_nvof {
        public:
            nvof_t* of;

            c_nvof(int width, int height) {
                of = nvof_create(this, width, height);
                if (!of) {
                    throw std::runtime_error("Failed to create nvof_t");
                }
            }

            ~c_nvof() {
                if (of) {
                    nvof_destroy(of);
                }
            }

            std::tuple<py::array_t<uint8_t>, py::array_t<float>> run(std::shared_ptr<c_image> img) {
                image_t* raw = img->raw();
                nvof_results_t* result = nvof_execute(of, raw);

                if (!result || !result->costs || !result->flow) {
                    throw std::runtime_error("nvof_execute failed or returned null results");
                }

                int h = result->grid_h;
                int w = result->grid_w;

                // Wrap costs as uint8 NumPy array
                auto costs = py::array_t<uint8_t>({h, w}, result->costs);

                // Wrap flow as (h, w, 2) float NumPy array
                auto flow = py::array_t<float, py::array::c_style>({h, w, 2});

                auto flow_buf = flow.mutable_unchecked<3>();
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        int idx = y * w + x;
                        flow_buf(y, x, 0) = result->flow[idx].flowx/(4*32.0*w);
                        flow_buf(y, x, 1) = result->flow[idx].flowy/(4*32.0*h);
                    }
                }

                return std::make_tuple(costs, flow);
            }
        };

class c_pcap_decoder {
    public:
        pcap_decoder_t* dec;

        c_pcap_decoder(const std::string& filename) {
            dec = pcap_decoder_create(filename.c_str());
            if (!dec) {
                throw std::runtime_error("Failed to create pcap_decoder");
            }
        }

        ~c_pcap_decoder() {
            if (dec) {
                pcap_decoder_destroy(dec);
            }
        }

        std::shared_ptr<c_image> get_frame() {
            image_t* img = pcap_decoder_get_frame(dec);
            if (!img) return nullptr;
            return std::make_shared<c_image>(img);
        }
    };

class PyTrackSharedState {
public:
    explicit PyTrackSharedState(const std::string& config_path) {
        state = track_shared_state_create(config_path.c_str());
        if (!state)
            throw std::runtime_error("Failed to create shared state");
    }

    ~PyTrackSharedState() {
        if (state)
            track_shared_state_destroy(state);
    }

    py::dict get_model_description() {
        return  convert_model_description(track_shared_state_get_model_description(state));
    }

    track_shared_state_t* get() const { return state; }

private:
    track_shared_state_t* state = nullptr;
};

class PyTrackStream {
public:
    explicit PyTrackStream(std::shared_ptr<PyTrackSharedState> shared_state_py)
        : shared_state(shared_state_py)
    {
        stream = track_stream_create(shared_state->get(), nullptr, nullptr);
        if (!stream)
            throw std::runtime_error("Failed to create track stream");
    }

    ~PyTrackStream() {
        if (stream)
            track_stream_destroy(stream);
    }

    void set_frame_intervals(double min_process, double min_full_roi) {
        track_stream_set_minimum_frame_intervals(stream, min_process, min_full_roi);
    }

    void run_on_images(const std::vector<std::shared_ptr<c_image>>& images) {
        for (const auto& img : images)
            track_stream_run_frame_time(stream, img->raw());
    }

    void run_on_video_file(const char *file, simple_decoder_codec_t codec, double video_fps) {
        track_stream_run_video_file(stream, file, codec, video_fps);
    }

    std::vector<py::dict> get_results() {
        std::vector<py::dict> results_out;
        auto results = track_stream_get_results(stream);
        for (auto& res : results) {
            py::dict d;
            d["result_type"] = res.result_type;
            d["time"] = res.time;
            d["motion_roi"] = convert_roi(res.motion_roi);
            d["inference_roi"] = convert_roi(res.inference_roi);
            d["track_dets"] = convert_detections(res.track_dets);
            d["inference_dets"] = convert_detections(res.inference_dets);
            results_out.push_back(d);
        }
        return results_out;
    }

private:
    track_stream_t* stream = nullptr;
    std::shared_ptr<PyTrackSharedState> shared_state;  // Shared ownership to keep it alive
};

class c_motion_tracker {
public:
    motion_track_t* mt;

    c_motion_tracker(const std::string& config_yaml) {
        mt = motion_track_create(config_yaml.c_str());
        if (!mt) {
            throw std::runtime_error("Failed to create motion_track_t");
        }
    }

    ~c_motion_tracker() {
        if (mt) {
            motion_track_destroy(mt);
        }
    }

    void add_frame(std::shared_ptr<c_image> img) {
        motion_track_add_frame(mt, img->raw());
    }

    std::vector<float> get_roi() {
        roi_t roi = motion_track_get_roi(mt);
        return std::vector<float>(roi.box, roi.box + 4);
    }

    void set_roi(const std::vector<float>& roi_vec) {
        if (roi_vec.size() != 4) {
            throw std::runtime_error("ROI must be a list of 4 floats");
        }
        roi_t roi;
        for (int i = 0; i < 4; ++i) {
            roi.box[i] = roi_vec[i];
        }
        motion_track_set_roi(mt, roi);
    }
};

PYBIND11_MODULE(ubon_pycstuff, m) {
    //std::cout << "ubon_pycstuff bindings loaded for version" << ubon_cstuff_get_version() << std::endl;
    py::enum_<image_format>(m, "ImageFormat")
        .value("NONE", IMAGE_FORMAT_NONE)
        .value("YUV420_DEVICE", IMAGE_FORMAT_YUV420_DEVICE)
        .value("YUV420_HOST", IMAGE_FORMAT_YUV420_HOST)
        .value("NV12_DEVICE", IMAGE_FORMAT_NV12_DEVICE)
        .value("RGB24_HOST", IMAGE_FORMAT_RGB24_HOST)
        .value("RGB24_DEVICE", IMAGE_FORMAT_RGB24_DEVICE)
        .value("RGB_PLANAR_FP16_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)
        .value("RGB_PLANAR_FP16_HOST", IMAGE_FORMAT_RGB_PLANAR_FP16_HOST)
        .value("RGB_PLANAR_FP32_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)
        .value("RGB_PLANAR_FP32_HOST", IMAGE_FORMAT_RGB_PLANAR_FP32_HOST)
        .value("MONO_HOST", IMAGE_FORMAT_MONO_HOST)
        .value("MONO_DEVICE", IMAGE_FORMAT_MONO_DEVICE)
        .export_values();

    py::enum_<match_type>(m, "MatchType")
        .value("MATCH_TYPE_BOX_IOU", MATCH_TYPE_BOX_IOU)
        .value("MATCH_TYPE_FACE_KP", MATCH_TYPE_FACE_KP)
        .value("MATCH_TYPE_POSE_KP", MATCH_TYPE_POSE_KP)
        .export_values();

    py::class_<c_image, std::shared_ptr<c_image>>(m, "c_image")
        .def(py::init<int, int, image_format_t>(), "Create empty image")
        .def_static("from_numpy", &c_image::from_numpy, "Create from NumPy RGB array")
        .def_property_readonly("size", &c_image::size, "Return (height, width) of the image")
        .def_property_readonly("format", &c_image::format, "Get the image format")
        .def_property_readonly("format_name", &c_image::format_name, "Get the image format name as a string")
        .def_property_readonly("timestamp", &c_image::timestamp, "Get the image timestamp")
        .def("__repr__",
            [](const c_image &self) {
                std::ostringstream oss;
                oss << "<c_image format='" << image_format_name(self.format())
                    << "' width:" << self.size().second << ", height:" << self.size().first << ")>";
                return oss.str();
            }
        )
        .def("scale", &c_image::scale, "Scale image")
        .def("convert", &c_image::convert, "Convert format")
        .def("to_numpy", &c_image::to_numpy, "Get NumPy RGB image")
        .def("display", &c_image::display, "Show image in a debug display")
        .def("hash", &c_image::hash, "Return hash of imge data")
        .def("sync", &c_image::sync, "Wait for all outstanding image ops")
        .def("blur", &c_image::blur, "Return gaussian blur of image")
        .def("mad_4x4", &c_image::mad_4x4, "Return 4x4 MAD of source images")
        .def("crop", &c_image::crop, "Return a crop of the surface")
        .def("blend", &c_image::blend, "Blend a rectange from a second surface over the top");

    m.def("c_get_aligned_faces", [](const std::vector<std::shared_ptr<c_image>>& images,
                                  const std::vector<float>& face_points,
                                  int width,
                                  int height) {
            // Number of images / faces
            size_t n = images.size();
            if (face_points.size() != 10 * n) {
                throw std::runtime_error("face_points must be a flat list of length 2*5 * n");
            }

            // Prepare input image_t* array
            std::vector<image_t*> in_ptrs;
            in_ptrs.reserve(n);
            for (const auto& img : images) {
                img->sync();
                in_ptrs.push_back(img->raw());
            }

            // Prepare output array
            std::vector<image_t*> out_ptrs(n);

            // Call C function: face_points.data() is already flat floats
            image_get_aligned_faces(in_ptrs.data(), const_cast<float*>(face_points.data()),
                                    static_cast<int>(n), width, height, out_ptrs.data());

            // Wrap outputs in shared_ptr<c_image>
            std::vector<std::shared_ptr<c_image>> results;
            results.reserve(n);
            for (auto ptr : out_ptrs) {
                results.emplace_back(std::make_shared<c_image>(ptr));
            }
            return results;
        },
        py::arg("images"),
        py::arg("face_points"),
        py::arg("width"),
        py::arg("height"),
        "Align faces in images given flat list of face landmarks and return list of aligned c_image objects.");

    py::class_<c_infer, std::shared_ptr<c_infer>>(m, "c_infer")
        .def(py::init<const std::string&, const std::string&>(), py::arg("trt_file"), py::arg("yaml_file"))
        .def("run", &c_infer::run, "Run inference on a c_image")
        .def("run_batch", &c_infer::run_batch, py::arg("images"), "Run batched inference on a list of c_image")
        .def("configure", &c_infer::configure, py::arg("config_dict"), "Configure inference parameters from a dictionary")
        .def("get_model_description", &c_infer::get_model_description, "Get model description as dictionary");

    py::class_<c_infer_thread, std::shared_ptr<c_infer_thread>>(m, "c_infer_thread")
        .def(py::init<const std::string&, const std::string&, py::dict>(), py::arg("trt_file"), py::arg("yaml_file"), py::arg("config_dict"))
        .def("infer_async", &c_infer_thread::infer_async, py::arg("image"), py::arg("roi"), "Submit async inference")
        .def("wait", &c_infer_thread::wait, py::arg("result_handle"), "Wait for result and get detections + timings")
        .def("get_stats", &c_infer_thread::get_stats, "Get internal inference stats")
        .def("get_model_description", &c_infer_thread::get_model_description, "Get model description");

    py::class_<c_infer_aux, std::shared_ptr<c_infer_aux>>(m, "c_infer_aux")
        .def(py::init<const std::string&>(), py::arg("trt_file"))
        .def("run", &c_infer_aux::run,
            py::arg("image"), py::arg("keypoints") = py::none(),
            "Run auxiliary inference on a single image and optional keypoints (keypoints=None=>images already aligned)")
        .def("run_batch", &c_infer_aux::run_batch,
            py::arg("images"), py::arg("keypoints") = py::none(),
            "Run auxiliary inference on a batch of images and optional keypoints (keypoints=None=>images already aligned)")
        .def("get_model_description", &c_infer_aux::get_model_description,
            "Get auxiliary model description as dictionary");

    py::enum_<result_type_t>(m, "ResultType")
        .value("TRACK_FRAME_SKIP_FRAMERATE", TRACK_FRAME_SKIP_FRAMERATE)
        .value("TRACK_FRAME_SKIP_NO_MOTION", TRACK_FRAME_SKIP_NO_MOTION)
        .value("TRACK_FRAME_SKIP_NO_IMG", TRACK_FRAME_SKIP_NO_IMG)
        .value("TRACK_FRAME_TRACKED_ROI", TRACK_FRAME_TRACKED_ROI)
        .value("TRACK_FRAME_TRACKED_FULL_REFRESH", TRACK_FRAME_TRACKED_FULL_REFRESH)
        .export_values();

    py::class_<PyTrackSharedState, std::shared_ptr<PyTrackSharedState>>(m, "c_track_shared_state")
        .def(py::init<const std::string&>())
        .def("get_model_description", &PyTrackSharedState::get_model_description);

    py::class_<PyTrackStream>(m, "c_track_stream")
        .def(py::init<std::shared_ptr<PyTrackSharedState>>(), py::arg("shared_state"))
        .def("set_frame_intervals", &PyTrackStream::set_frame_intervals)
        .def("run_on_images", &PyTrackStream::run_on_images)
        .def("run_on_video_file", &PyTrackStream::run_on_video_file)
        .def("get_results", &PyTrackStream::get_results);

    py::enum_<simple_decoder_codec_t>(m, "SimpleDecoderCodec")
        .value("SIMPLE_DECODER_CODEC_H264", SIMPLE_DECODER_CODEC_H264)
        .value("SIMPLE_DECODER_CODEC_H265", SIMPLE_DECODER_CODEC_H265)
        .export_values();

    py::class_<c_decoder, std::shared_ptr<c_decoder>>(m, "c_decoder")
        .def(py::init<simple_decoder_codec_t>())
        .def("decode", &c_decoder::decode, py::arg("bitstream"));

    py::class_<c_nvof, std::shared_ptr<c_nvof>>(m, "c_nvof")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("run", &c_nvof::run, "Run NVIDIA Optical Flow on a c_image");

    py::class_<c_pcap_decoder, std::shared_ptr<c_pcap_decoder>>(m, "c_pcap_decoder")
        .def(py::init<const std::string&>(), py::arg("filename"))
        .def("get_frame", &c_pcap_decoder::get_frame, "Get the next decoded frame (returns c_image or None)");

    py::class_<c_motion_tracker, std::shared_ptr<c_motion_tracker>>(m, "c_motion_tracker")
        .def(py::init<const std::string&>(), py::arg("config_yaml"))
        .def("add_frame", &c_motion_tracker::add_frame, "Add image frame to tracker")
        .def("get_roi", &c_motion_tracker::get_roi, "Get current ROI as list of 4 floats")
        .def("set_roi", &c_motion_tracker::set_roi, py::arg("roi"), "Set ROI from list of 4 floats");

    m.def("load_jpeg", &c_load_jpeg,
          "load jpeg file to c img",
          py::arg("file"));

    m.def("match_box_iou", &match_box_iou_wrapper,
          "Matches boxes using IoU and returns index lists",
          py::arg("dets_a"), py::arg("dets_b"), py::arg("iou_thr"), py::arg("match_type"));

    m.def("cuda_set_sync_mode", &cuda_set_sync_mode,
            py::arg("force_sync"),
            py::arg("force_default_stream"),
            "Sets CUDA synchronization mode. "
            "`force_sync`: forces synchronization after operations. "
            "`force_default_stream`: restricts to the default stream.");

    m.def("log_set_level", &log_set_level,
            py::arg("level"),
            "Set the logging level. "
            "`level`: 0 (TRACE) to 5 (FATAL).");

    m.doc() = "ubon_cstuff python module";
    m.def("get_version", ubon_cstuff_get_version, "Returns the git version of the library");

    log_set_level(LOG_WARN);
    init_cuda_stuff();
    image_init();
}

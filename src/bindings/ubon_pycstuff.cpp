#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;  // <-- this line enables "_a" syntax

#include "image.h"
#include "infer.h"
#include "cuda_stuff.h"
#include "detections.h"
#include "simple_decoder.h"
#include "display.h"
#include "nvof.h"
#include "log.h"
#include "misc.h"
#include "pcap_decoder.h"

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
        auto buf = input_rgb.unchecked<3>();
        int height = buf.shape(0);
        int width = buf.shape(1);

        image_t* img = create_image(width, height, IMAGE_FORMAT_RGB24_HOST);

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    img->rgb[3 * x + c + img->stride_rgb * y] = buf(y, x, c);

        return std::make_shared<c_image>(img);
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
            image_sync(tmp);
            auto result = py::array_t<uint8_t>({tmp->height, tmp->width, 3});
            auto buf = result.mutable_unchecked<3>();

            for (int y = 0; y < tmp->height; ++y)
                for (int x = 0; x < tmp->width; ++x)
                    for (int c = 0; c < 3; ++c)
                        buf(y, x, c) = tmp->rgb[3 * x + c + tmp->stride_rgb * y];

            destroy_image(tmp);
            return result;
        }
    }

    image_t* raw() { return img; }
};

class c_infer {
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
        detections_t* dets = infer(inf, img);

        py::list results;
        for (int i = 0; i < dets->num_detections; ++i) {
            detection_t* det = &dets->det[i];

            py::dict item;
            item["cls"] = det->cl;
            item["conf"] = det->conf;
            py::list box;
            box.append(det->x0);
            box.append(det->y0);
            box.append(det->x1);
            box.append(det->y1);
            item["box"] = box;
            results.append(item);
        }

        destroy_detections(dets);
        return results;
    }

    infer_t* raw() { return inf; }
};

class c_decoder {
    public:
        simple_decoder_t* dec;
        std::vector<std::shared_ptr<c_image>> current_output;

        c_decoder() {
            // Create the decoder, register our static callback, pass `this` as context
            dec = simple_decoder_create(this, &c_decoder::on_frame_static, SIMPLE_DECODER_CODEC_H264);
            if (!dec) {
                throw std::runtime_error("Failed to create decoder");
            }
        }

        ~c_decoder() {
            if (dec) simple_decoder_destroy(dec);
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

    py::class_<c_infer, std::shared_ptr<c_infer>>(m, "c_infer")
        .def(py::init<const std::string&, const std::string&>(), py::arg("trt_file"), py::arg("yaml_file"))
        .def("run", &c_infer::run, "Run inference on a c_image");

    py::class_<c_decoder, std::shared_ptr<c_decoder>>(m, "c_decoder")
        .def(py::init<>())
        .def("decode", &c_decoder::decode, py::arg("bitstream"));

    py::class_<c_nvof, std::shared_ptr<c_nvof>>(m, "c_nvof")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("run", &c_nvof::run, "Run NVIDIA Optical Flow on a c_image");

    py::class_<c_pcap_decoder, std::shared_ptr<c_pcap_decoder>>(m, "c_pcap_decoder")
        .def(py::init<const std::string&>(), py::arg("filename"))
        .def("get_frame", &c_pcap_decoder::get_frame, "Get the next decoded frame (returns c_image or None)");

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

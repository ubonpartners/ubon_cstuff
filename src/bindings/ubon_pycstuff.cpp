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
    
        py::array_t<uint8_t> to_numpy() {
            image_t* tmp = (img->format == IMAGE_FORMAT_RGB24_HOST)
                ? image_reference(img)
                : image_convert(img, IMAGE_FORMAT_RGB24_HOST);
    
            auto result = py::array_t<uint8_t>({tmp->height, tmp->width, 3});
            auto buf = result.mutable_unchecked<3>();
    
            for (int y = 0; y < tmp->height; ++y)
                for (int x = 0; x < tmp->width; ++x)
                    for (int c = 0; c < 3; ++c)
                        buf(y, x, c) = tmp->rgb[3 * x + c + tmp->stride_rgb * y];
    
            destroy_image(tmp);
            return result;
        }
    
        image_t* raw() { return img; }
    };

/*static py::capsule py_create_image(int width, int height, image_format_t fmt) 
{
    image_t *img=create_image(width, height, fmt);
    return py::capsule(img, "image_t", [](PyObject* capsule) {
        auto ptr = reinterpret_cast<image_t*>(PyCapsule_GetPointer(capsule, "image_t"));
        delete ptr;
    });
}

static void py_destroy_image(py::capsule handle) {
    if (std::string(handle.name()) != "image_t")
        throw std::runtime_error("Invalid capsule type expected img_t ");
    image_t* img = reinterpret_cast<image_t*>(handle.get_pointer());
    destroy_image(img);
}

static void py_image_reference(py::capsule handle) {
    if (std::string(handle.name()) != "image_t")
        throw std::runtime_error("Invalid capsule type expected img_t");
    image_t* img = reinterpret_cast<image_t*>(handle.get_pointer());
    image_reference(img);
}

static py::capsule py_image_scale(py::capsule handle, int width, int height) {
    if (std::string(handle.name()) != "image_t")
        throw std::runtime_error("Invalid capsule type expected img_t");
    image_t* img = reinterpret_cast<image_t*>(handle.get_pointer());
    image_t* scaled = image_scale(img, width, height);
    printf("scaled=%p\n",scaled);
    return py::capsule(scaled, "image_t", [](PyObject* capsule) {
        auto ptr = reinterpret_cast<image_t*>(PyCapsule_GetPointer(capsule, "image_t"));
        delete ptr;
    });
}*/


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
                dec = simple_decoder_create(this, &c_decoder::on_frame_static);
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

PYBIND11_MODULE(ubon_pycstuff, m) {
    std::cout << "ubon_pycstuff bindings loaded" << std::endl;
    py::enum_<image_format>(m, "ImageFormat")
        .value("YUV420_DEVICE", IMAGE_FORMAT_YUV420_DEVICE)
        .value("YUV420_HOST", IMAGE_FORMAT_YUV420_HOST)
        .value("NV12_DEVICE", IMAGE_FORMAT_NV12_DEVICE)
        .value("RGB24_HOST", IMAGE_FORMAT_RGB24_HOST)
        .value("RGB24_DEVICE", IMAGE_FORMAT_RGB24_DEVICE)
        .value("RGB_PLANAR_FP16_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)
        .value("RGB_PLANAR_FP16_HOST", IMAGE_FORMAT_RGB_PLANAR_FP16_HOST)
        .value("RGB_PLANAR_FP32_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)
        .value("RGB_PLANAR_FP32_HOST", IMAGE_FORMAT_RGB_PLANAR_FP32_HOST)
        .export_values();

    py::class_<c_image, std::shared_ptr<c_image>>(m, "c_image")
        .def(py::init<int, int, image_format_t>(), "Create empty image")
        .def_static("from_numpy", &c_image::from_numpy, "Create from NumPy RGB array")
        .def("scale", &c_image::scale, "Scale image")
        .def("convert", &c_image::convert, "Convert format")
        .def("to_numpy", &c_image::to_numpy, "Get NumPy RGB image")
        .def("display", &c_image::display, "Show image in a debug display");

    py::class_<c_infer, std::shared_ptr<c_infer>>(m, "c_infer")
        .def(py::init<const std::string&, const std::string&>(), py::arg("trt_file"), py::arg("yaml_file"))
        .def("run", &c_infer::run, "Run inference on a c_image");

    py::class_<c_decoder, std::shared_ptr<c_decoder>>(m, "c_decoder")
        .def(py::init<>())
        .def("decode", &c_decoder::decode, py::arg("bitstream"));

    py::class_<c_nvof, std::shared_ptr<c_nvof>>(m, "c_nvof")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("run", &c_nvof::run, "Run NVIDIA Optical Flow on a c_image");

    init_cuda_stuff();
}
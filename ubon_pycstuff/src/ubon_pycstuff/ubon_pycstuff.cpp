#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;  // <-- this line enables "_a" syntax

#include "image.h"
#include "infer.h"
#include "cuda_stuff.h"
#include "detections.h"

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


static py::capsule py_infer_create(const std::string& py_str) 
{
    infer_t *inf=infer_create(py_str.c_str());
    return py::capsule(inf, "infer_t", [](PyObject* capsule) {
        auto ptr = reinterpret_cast<image_t*>(PyCapsule_GetPointer(capsule, "infer_t"));
        delete ptr;
    });
}

static void py_infer_destroy(py::capsule handle) {
    if (std::string(handle.name()) != "infer_t")
        throw std::runtime_error("Invalid capsule type expected infer_t ");
    infer_t *inf= reinterpret_cast<infer_t*>(handle.get_pointer());
    infer_destroy(inf);
}

static py::list py_infer(py::capsule inf_handle, std::shared_ptr<c_image> image_obj) {
    if (std::string(inf_handle.name()) != "infer_t")
        throw std::runtime_error("Invalid capsule type expected infer_t");

    infer_t* inf = reinterpret_cast<infer_t*>(inf_handle.get_pointer());
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

PYBIND11_MODULE(ubon_pycstuff, m) {
    py::enum_<image_format>(m, "ImageFormat")
        .value("UNKNOWN", IMAGE_FORMAT_YUV420_HOST)
        .value("YUV420_HOST", IMAGE_FORMAT_YUV420_DEVICE)
        .value("NV12_DEVICE", IMAGE_FORMAT_NV12_DEVICE)
        .value("RGB24_HOST", IMAGE_FORMAT_RGB24_HOST)
        .value("RGB24_DEVICE", IMAGE_FORMAT_RGB24_DEVICE)
        .value("RGB_PLANAR_FP16_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP16_DEVICE)
        .value("RGB_PLANAR_FP16_HOST", IMAGE_FORMAT_RGB_PLANAR_FP16_HOST)
        .value("RGB_PLANAR_FP32_HOST", IMAGE_FORMAT_RGB_PLANAR_FP32_HOST)
        .value("RGB_PLANAR_FP32_DEVICE", IMAGE_FORMAT_RGB_PLANAR_FP32_DEVICE)
        .export_values();

    py::class_<c_image, std::shared_ptr<c_image>>(m, "c_image")
        .def(py::init<int, int, image_format_t>(), "Create empty image")
        .def_static("from_numpy", &c_image::from_numpy, "Create from NumPy RGB array")
        .def("scale", &c_image::scale, "Scale image")
        .def("convert", &c_image::convert, "Convert format")
        .def("to_numpy", &c_image::to_numpy, "Get NumPy RGB image");

    m.def("create_image", &py_create_image, "Create image", py::arg("width"), py::arg("height"), py::arg("fmt"));
    m.def("destroy_image", &py_destroy_image, "Destroy image", py::arg("handle"));
    m.def("image_reference", &py_image_reference, "image reference", py::arg("handle"));
    m.def("image_scale", &py_image_scale, "scale image", py::arg("handle"), py::arg("width"), py::arg("height"));
    m.def("image_convert", &py_image_convert, "image_convert", py::arg("handle"), py::arg("fmt"));
    m.def("create_image_from_np", &py_create_image_from_np, "Create image from NumPy RGB array");
    m.def("get_np_image", &py_get_np_image, "Extract NumPy RGB array from image");
    
    m.def("infer_create", &py_infer_create, "Create tensorRT inference object", py::arg("trt_file"));
    m.def("infer_destroy", &py_infer_destroy, "Destroy tensorRT inference object", py::arg("handle"));
    m.def("infer", &py_infer, "Run inference", py::arg("inf_handle"), py::arg("surf_handle"));

    init_cuda_stuff();
}
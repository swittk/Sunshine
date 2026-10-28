/**
 * QEMU Display1 capture source for Sunshine.
 *
 * Isolated experimental adapter. Depends on qmdp's tested D-Bus receiver,
 * not on a host Linux compositor or a new graphics device in the VM.
 * This backend reads QEMU CPU scanout frames; it is NOT a zero-copy encoder.
 */
#include "src/platform/common.h"
#include "src/video.h"
#include "src/logging.h"
#ifdef SUNSHINE_BUILD_CUDA
#include "cuda.h"
#endif
#ifdef SUNSHINE_BUILD_VAAPI
#include "vaapi.h"
#endif

#include "capture_feed.hpp"
#include "input_route.hpp"
#include "pulse_audio_bridge.hpp"
#include "gio_qemu_display.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace platf {
namespace {
using namespace std::chrono_literals;

class qemu_img_t final: public img_t {
public:
    std::vector<std::uint8_t> buffer;
};

class qemu_display_t final: public display_t {
public:
    qemu_display_t(mem_type_e mem_type, const std::string &address,
                   const std::string &destination):
        mem_type_(mem_type),
        qemu_(std::make_shared<reims_stream::GioQemuDisplay>(address,destination,use_guest_audio())),
        feed_(*qemu_) {
        feed_.set_audio_callback([this](std::span<const float> data,std::uint32_t rate,std::uint16_t channels){
            audio_.submit(data,rate,channels);
        });
    }

    ~qemu_display_t() override {
        reims_stream::InputRoute::instance().detach(qemu_.get());
        feed_.stop();
        audio_.stop();
    }

    bool initialize() {
        if (use_guest_audio()) audio_.start();
        feed_.start();
        auto geometry = feed_.wait_geometry(3000ms);
        if (!geometry) return false;
        if (geometry->width == 0U || geometry->height == 0U ||
            geometry->width > 16384U || geometry->height > 16384U) return false;
        width = env_width = logical_width = env_logical_width = geometry->width;
        height = env_height = logical_height = env_logical_height = geometry->height;
        reims_stream::InputRoute::instance().attach(qemu_);
        return true;
    }

    capture_e capture(const push_captured_image_cb_t &push,
                      const pull_free_image_cb_t &pull, bool * /*cursor*/) override {
        for (;;) {
            std::shared_ptr<img_t> img;
            if (!pull(img)) return capture_e::interrupted;
            if (!img) return capture_e::capture_error;
            auto *out = static_cast<qemu_img_t *>(img.get());
            const std::size_t size = static_cast<std::size_t>(width) * height * 4U;
            if (out->buffer.size() < size) out->buffer.resize(size);
            reims_stream::FrameMetadata metadata;
            const auto result = feed_.next_bgra(
                std::span<std::uint8_t>(out->buffer.data(),size),
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                500ms, metadata);
            switch (result) {
                case reims_stream::CaptureStatus::frame:
                    img->data = out->buffer.data();
                    img->row_pitch = width * 4;
                    img->pixel_pitch = 4;
                    img->width = width;
                    img->height = height;
                    img->frame_timestamp = metadata.produced_at;
                    if (!push(std::move(img),true)) return capture_e::ok;
                    break;
                case reims_stream::CaptureStatus::timeout:
                    if (!push(std::move(img),false)) return capture_e::ok;
                    break;
                case reims_stream::CaptureStatus::resize:
                    return capture_e::reinit;
                case reims_stream::CaptureStatus::interrupted:
                    return capture_e::interrupted;
                case reims_stream::CaptureStatus::error:
                    return capture_e::capture_error;
            }
        }
    }

    std::shared_ptr<img_t> alloc_img() override {
        auto img = std::make_shared<qemu_img_t>();
        img->buffer.resize(static_cast<std::size_t>(width)*height*4U);
        img->data = img->buffer.data();
        img->width = width;
        img->height = height;
        img->pixel_pitch = 4;
        img->row_pitch = width*4;
        return img;
    }

    int dummy_img(img_t *img) override {
        if (!img) return -1;
        auto *frame = static_cast<qemu_img_t *>(img);
        std::fill(frame->buffer.begin(),frame->buffer.end(),0);
        img->data = frame->buffer.data();
        img->width = width;
        img->height = height;
        img->pixel_pitch = 4;
        img->row_pitch = width*4;
        return 0;
    }

    std::unique_ptr<avcodec_encode_device_t> make_avcodec_encode_device(pix_fmt_e pixfmt) override {
#ifdef SUNSHINE_BUILD_CUDA
        if (mem_type_ == mem_type_e::cuda)
            return cuda::make_avcodec_encode_device(width,height,false);
#endif
#ifdef SUNSHINE_BUILD_VAAPI
        if (mem_type_ == mem_type_e::vaapi)
            return va::make_avcodec_encode_device(width,height,false);
#endif
        return std::make_unique<avcodec_encode_device_t>();
    }

private:
    static bool use_guest_audio() {
        const char *setting=std::getenv("REIMS_ENABLE_AUDIO");
        return setting && std::strcmp(setting,"1")==0;
    }
    mem_type_e mem_type_;
    std::shared_ptr<reims_stream::GioQemuDisplay> qemu_;
    reims_stream::PulseAudioBridge audio_;
    reims_stream::CaptureFeed feed_;
};
} // namespace

std::vector<std::string> qemu_dbus_display_names() {
    const char *address = std::getenv("QEMU_DBUS_ADDRESS");
    return address && address[0] ? std::vector<std::string>{"QEMU-0"} : std::vector<std::string>{};
}

std::shared_ptr<display_t> qemu_dbus_display(mem_type_e mem_type,
                                              const std::string &,
                                              const video::config_t &) {
    const char *address = std::getenv("QEMU_DBUS_ADDRESS");
    if (!address || !address[0]) {
        BOOST_LOG(error) << "Set QEMU_DBUS_ADDRESS for QEMU capture";
        return nullptr;
    }
#ifdef SUNSHINE_BUILD_CUDA
    if (mem_type == mem_type_e::cuda && cuda::init() != 0) {
        BOOST_LOG(error) << "Cannot initialize CUDA for QEMU capture";
        return nullptr;
    }
#endif
    const char *dest = std::getenv("QEMU_DBUS_DESTINATION");
    // broker:/path and inherited fd: modes use an empty destination.
    // -display dbus,addr=... connects to a message bus and uses org.qemu.
    auto backend = std::make_shared<qemu_display_t>(
        mem_type, address, dest ? std::string(dest) : std::string("org.qemu"));
    try {
        if (!backend->initialize()) return nullptr;
        return backend;
    } catch (const std::exception &e) {
        BOOST_LOG(error) << "QEMU D-Bus capture initialization failed: " << e.what();
        return nullptr;
    }
}
} // namespace platf

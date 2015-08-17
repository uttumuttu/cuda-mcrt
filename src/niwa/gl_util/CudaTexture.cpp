#include "CudaTexture.h"

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

namespace niwa {
    namespace gl_util {
        CudaTexture::CudaTexture(size_t width, size_t height) 
            : width_(width), height_(height), pbo_(NULL), texture_(NULL) {
            glGenBuffers(1, &pbo_);

            // set the PBO the current unpack buffer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

            // allocate data for the buffer, 4-channel 8-bit image
            size_t nTexels = width_ * height_;
            size_t nValues = nTexels * 4;
            size_t nBytes  = sizeof(GLubyte) * nValues;

            glBufferData(GL_PIXEL_UNPACK_BUFFER, nBytes, NULL, GL_DYNAMIC_COPY);

            cudaGLRegisterBufferObject(pbo_);

            // TODO: how to "unbind" buffer?

            glEnable(GL_TEXTURE_2D);
            glGenTextures(1, &texture_);
            glBindTexture(GL_TEXTURE_2D, texture_);

            // allocate texture memory
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            // N.B.: GL_TEXTURE_RECTANGLE_ARGB may be used instead of
            //       GL_TEXTURE_2D for improved performance if linear interpolation
            //       is not desired
        }

        CudaTexture::~CudaTexture() {
            glDeleteTextures(1, &texture_);
            texture_ = NULL;

            cudaGLUnregisterBufferObject(pbo_);

            // TODO: is this bind really required?
            //       no wai
            glBindBuffer(GL_ARRAY_BUFFER, pbo_);

            glDeleteBuffers(1, &pbo_);

            pbo_ = NULL;
        }

        size_t const CudaTexture::width() const {
            return width_;
        }

        size_t const CudaTexture::height() const {
            return height_;
        }

        void CudaTexture::render(IRenderCallback const& callback) const {
            uchar4* dptr = NULL;

            // Map OpenGL buffer object for writing from CUDA on a single GPU.
            // no data is moved. When mapped to CUDA, OpenGL should NOT
            // use this buffer.
            cudaGLMapBufferObject((void**)&dptr, pbo_);

            callback.render(dptr);

            cudaGLUnmapBufferObject(pbo_);
        }

        CudaTexture::IRenderCallback::~IRenderCallback() {
            // ignored
        }

        void CudaTexture::bind() const {
            // create a texture from the buffer
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);

            glBindTexture(GL_TEXTURE_2D, texture_);

            // glTexSubImage2D is required for format-conversion (in case they differ)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        }

        void CudaTexture::unbind() const {
            // TODO
        }
    }
}

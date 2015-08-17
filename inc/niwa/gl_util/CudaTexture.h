#ifndef NIWA_GL_UTIL_CUDATEXTURE_H
#define NIWA_GL_UTIL_CUDATEXTURE_H

#include <gl/glew.h>
#include <cuda_runtime.h>

namespace niwa {
    namespace gl_util {
        /**
         * A texture that can be drawn into
         * with CUDA.
         */
        class CudaTexture {
        public:
            class IRenderCallback {
            public:
                virtual ~IRenderCallback();

                virtual void render(uchar4* data) const = 0;
            };

        public:
            CudaTexture(size_t width, size_t height);

            ~CudaTexture();

        public:
            size_t const width() const;
            size_t const height() const;

        public:
            /** 
             * Renders to the texture via the callback. Must be called before bind and unbind.
             */
            void render(IRenderCallback const& callback) const;

            /** 
             * Binds the texture so it can be drawn on the screen;
             * must be called after render, before bind.
             */
            void bind() const;

            /**
             * Unbinds the texture after it has been drawn on the screen;
             * must be called after render and bind.
             */
            void unbind() const;

        private:
            CudaTexture(CudaTexture const&);
            CudaTexture& operator = (CudaTexture const&);

        private:
            size_t const width_;
            size_t const height_;

            GLuint pbo_;
            GLuint texture_;
        };
    }
}

#endif
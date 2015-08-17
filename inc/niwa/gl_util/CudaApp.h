#ifndef NIWA_GL_UTIL_CUDAAPP_H
#define NIWA_GL_UTIL_CUDAAPP_H

#include <string>

namespace niwa {
    namespace gl_util {
        class IGlutCallbacks;

        class CudaApp {
        public:
            /**
             * @param argv Must have a whole-application lifetime.
             */
            CudaApp(int argc, char** argv, size_t width, size_t height, std::string const& title);
            ~CudaApp();

            void run(IGlutCallbacks& callbacks) const;

        private:
            std::string title_;
        };
    }
}

#endif
#ifndef NIWA_GL_UTIL_IGLUTCALLBACKS_H
#define NIWA_GL_UTIL_IGLUTCALLBACKS_H

namespace niwa {
    namespace gl_util {
        class IGlutCallbacks {
        public:
            virtual ~IGlutCallbacks();

            virtual void display() const = 0;

            virtual void keyboard(unsigned char key, int x, int y) = 0;

            virtual void mouse(int button, int state, int x, int y) = 0;

            virtual void motion(int x, int y) = 0;
        };
    }
}

#endif
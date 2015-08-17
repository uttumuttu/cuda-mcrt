#include "CudaApp.h"

#include "IGlutCallbacks.h"

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

namespace {
    static std::string GlobalTitle;

    static niwa::gl_util::IGlutCallbacks* GlobalCallbacks = NULL;

    static unsigned int GlobalTimer = 0; // a timer for FPS calculations

    static void computeFPS() {
        static int fpsCount = 0;
        static int fpsLimit = 5;

        ++fpsCount;

        if(fpsCount == fpsLimit) {
            char fps[256];
            float ifps = 1.f / (cutGetAverageTimerValue(GlobalTimer) / 1000.f);
            sprintf(fps, "%s: %5.1f fps", GlobalTitle.c_str(), ifps);

            glutSetWindowTitle(fps);
            fpsCount = 0;

            cutilCheckError(cutResetTimer(GlobalTimer));
        }
    }

    static void display() {
        cutilCheckError(cutStartTimer(GlobalTimer));

        if(GlobalCallbacks) {
            GlobalCallbacks->display();
        }

        cutilCheckError(cutStopTimer(GlobalTimer));
        computeFPS();
    }

    static void keyboard(unsigned char key, int x, int y) {
        if(GlobalCallbacks) {
            GlobalCallbacks->keyboard(key, x, y);
        }
    }

    static void mouse(int button, int state, int x, int y) {
        if(GlobalCallbacks) {
            GlobalCallbacks->mouse(button, state, x, y);
        }
    }

    static void motion(int x, int y) {
        if(GlobalCallbacks) {
            GlobalCallbacks->motion(x, y);
        }
    }

    static bool initGL(int argc, char** argv, size_t width, size_t height, std::string const& title) {
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

        //glutGameModeString("640x480:32");
        //glutEnterGameMode();

        glutInitWindowSize(width, height);
        glutCreateWindow(title.c_str());

        glewInit();
        if(!glewIsSupported("GL_VERSION_2_0")) {
            fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
            return false;
        }

        glViewport(0, 0, width, height);
    
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glDisable(GL_DEPTH_TEST);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

        return true;
    }
}

namespace niwa {
    namespace gl_util {
        CudaApp::CudaApp(int argc, char** argv, size_t width, size_t height, std::string const& title)
            : title_(title) {
            cutilCheckError(cutCreateTimer(&GlobalTimer));

            if (!initGL(argc, argv, width, height, title)) {
                exit(EXIT_FAILURE);
            }

            // First initialize OpenGL context, so we can properly set the
            // GL for CUDA. NVIDIA notes this is necessary in order to achieve
            // optimal performance with OpenGL/CUDA interop.
            if( cutCheckCmdLineFlag(argc, (const char **)argv, "device") ) {
                cutilGLDeviceInit(argc, argv);
            } else {
                // default
                cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
            }

            CUT_CHECK_ERROR_GL();
        }

        CudaApp::~CudaApp() {
            cudaThreadExit();
        }

        void CudaApp::run(IGlutCallbacks& callbacks) const {
            GlobalTitle = title_;

            GlobalCallbacks = &callbacks;

            glutDisplayFunc(display);
            glutKeyboardFunc(keyboard);
            glutMouseFunc(mouse);
            glutMotionFunc(motion);

            glutMainLoop();
        }
    }
}

#include "niwa/gl_util/CudaApp.h"
#include "niwa/gl_util/IGlutCallbacks.h"
#include "niwa/gl_util/CudaTexture.h"

#define NOMINMAX
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <rendercheck_gl.h>

#include "RandomState.h"
#include "Photonmap.h"

extern void raytracing(uchar4* color, DefaultPhotonMap const* photonMap,
                  RandomState* random, unsigned int image_width,
                  unsigned int image_height, float timeSeconds);

extern void photonmapping(
        int nPhotonPaths, DefaultPhotonList* lists,
        RandomState* rngs, float timeSeconds);

#define N_PHOTON_PATHS 0

namespace {
    class Callbacks : public niwa::gl_util::IGlutCallbacks {
    public:
        Callbacks(size_t width, size_t height) : texture_(width, height) {
            RandomState* rt_host = new RandomState[width * height];
            RandomState* pm_host = new RandomState[N_PHOTON_PATHS];

            for(size_t i=0; i<width*height; ++i) {
                rt_host[i].setSeed(rand());
            }
            for(size_t i=0; i<N_PHOTON_PATHS; ++i) {
                pm_host[i].setSeed(rand());
            }

            size_t nBytesRt = sizeof(RandomState) * width * height;
            size_t nBytesPm = sizeof(RandomState) * N_PHOTON_PATHS;

            cudaMalloc((void**) &rt_random_, nBytesRt);
            cudaMalloc((void**) &pm_random_, nBytesPm);
            cudaMemcpy(rt_random_, rt_host, nBytesRt, cudaMemcpyHostToDevice);
            cudaMemcpy(pm_random_, pm_host, nBytesPm, cudaMemcpyHostToDevice);

            cudaMalloc((void**) &lists_, sizeof(DefaultPhotonList) * N_PHOTON_PATHS);
            host_lists_ = new DefaultPhotonList[N_PHOTON_PATHS];

            cudaMalloc((void**) &device_map_, sizeof(DefaultPhotonMap));
            host_map_ = new DefaultPhotonMap;
        }

        ~Callbacks() {
            delete host_map_;
            delete[] host_lists_;

            cudaFree(device_map_);
            cudaFree(lists_);
            cudaFree(pm_random_);
            cudaFree(rt_random_);
        }

        void display() const {
            class Render : public niwa::gl_util::CudaTexture::IRenderCallback {
            public:
                Render(DefaultPhotonMap const* photonMap, RandomState* random, size_t width, size_t height, float timeSeconds)
                    : photonMap_(photonMap), random_(random), width_(width), height_(height), timeSeconds_(timeSeconds) {
                    // ignored
                }

                void render(uchar4* data) const {
                    raytracing(data, photonMap_, random_, width_, height_, timeSeconds_);
                }

            private:
                DefaultPhotonMap const* photonMap_;
                RandomState* random_;
                size_t width_;
                size_t height_;
                float timeSeconds_;
            };

            float timeSeconds = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

            //photonmapping(N_PHOTON_PATHS, lists_, pm_random_, timeSeconds);

            //cudaMemcpy(host_lists_, lists_, sizeof(DefaultPhotonList)*N_PHOTON_PATHS, cudaMemcpyDeviceToHost);

            //host_map_->construct(N_PHOTON_PATHS, host_lists_);

            //cudaMemcpy(device_map_, host_map_, sizeof(DefaultPhotonMap), cudaMemcpyHostToDevice);

            Render render(device_map_, rt_random_, texture_.width(), texture_.height(), timeSeconds);

            texture_.render(render);
            texture_.bind();

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 0.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 0.0f);
            glEnd();

            texture_.unbind();

            glutSwapBuffers();

            glutPostRedisplay(); // request redraw
        }

        void keyboard(unsigned char key, int x, int y) {
            switch(key) {
                case('q') :
                case(27) :
                    exit(0);
                    break;
                default:
                    break;
            }

            glutPostRedisplay();
        }

        void mouse(int button, int state, int x, int y) {
            // ignored
        }

        void motion(int x, int y) {
            // ignored
        }

    private:
        Callbacks(Callbacks const&);
        Callbacks& operator = (Callbacks const&);

    private:
        niwa::gl_util::CudaTexture texture_;

        DefaultPhotonList* lists_; // on-device
        DefaultPhotonList* host_lists_;

        DefaultPhotonMap* device_map_; 
        DefaultPhotonMap* host_map_;

        RandomState* rt_random_; // on-device
        RandomState* pm_random_; // on-device
    };
}

int main(int argc, char** argv) {
    niwa::gl_util::CudaApp app(argc, argv, 640, 480, "Ray tracing");

    Callbacks callbacks(640, 480);

    app.run(callbacks);

    cutilExit(argc, argv);
    return EXIT_SUCCESS;
}

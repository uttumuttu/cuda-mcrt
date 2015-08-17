#ifndef COMPOSITE_H
#define COMPOSITE_H

template <typename T1, typename T2>
class Composite {
public:
    __device__ Composite(T1 const& first, T2 const& second) : first_(first), second_(second) {
        // ignored
    }

    __device__ bool trace(Ray const& ray, HitInfo& info) const {
        HitInfo temp;

        bool hit1 = first_.trace(ray, info);
        bool hit2 = second_.trace(ray, temp);

        if(!hit1 | (hit2 & temp.time < info.time)) {
            info = temp;
        }
        return hit1 | hit2;
    }

    __device__ bool traceShadow(Ray const& ray, float distance) const {
        return first_.traceShadow(ray, distance)
            | second_.traceShadow(ray, distance);
    }

private:
    T1 const& first_;
    T2 const& second_;
};

template <typename T1, typename T2>
static __device__ Composite<T1,T2> cons(T1 const& first, T2 const& second) {
    return Composite<T1,T2>(first, second);
}

#endif
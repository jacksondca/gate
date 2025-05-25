#pragma once
#include "hnswlib.h"

namespace hnswlib {
static float
Cos(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2Sqr;
    auto ip_func = InnerProductDistance;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
CosSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD16ExtAVX512;
    auto ip_func = InnerProductDistanceSIMD16ExtAVX512;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
CosSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD16ExtAVX;
    auto ip_func = InnerProductDistanceSIMD16ExtAVX;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}

#endif

#if defined(USE_SSE)

static float
CosSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD16ExtSSE;
    auto ip_func = InnerProductDistanceSIMD16ExtSSE;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> CosSIMD16Ext = CosSIMD16ExtSSE;

static float
CosSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD16ExtResiduals;
    auto ip_func = InnerProductDistanceSIMD16ExtResiduals;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}
#endif

#if defined(USE_SSE)
static float
CosSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD4Ext;
    auto ip_func = InnerProductDistanceSIMD4Ext;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}

static float
CosSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    auto l2_func = L2SqrSIMD4ExtResiduals;
    auto ip_func = InnerProductDistanceSIMD4ExtResiduals;

    float ip = ip_func(pVect1v, pVect2v, qty_ptr);
    float dist = l2_func(pVect1v, pVect1v, qty_ptr) *
                    l2_func(pVect2v, pVect2v, qty_ptr);
    
    return 1. - ip / dist;
}
#endif

class CosSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    CosSpace(size_t dim) {
        fstdistfunc_ = Cos;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            CosSIMD16Ext = CosSIMD16ExtAVX512;
        else if (AVXCapable())
            CosSIMD16Ext = CosSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            CosSIMD16Ext = CosSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = CosSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = CosSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = CosSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = CosSIMD4ExtResiduals;
#endif

        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~CosSpace() {}
};
}
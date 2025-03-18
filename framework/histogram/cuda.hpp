#ifndef HISTOGRAM_CUDA_HPP
#define HISTOGRAM_CUDA_HPP

#include "interface.hpp"

#include "cuda/cuda.hpp"

#include <cstdint>
#include <vector>

template <typename T = char, typename RES = unsigned int>
class CudaHistogramAlgorithm : public IHistogramAlgorithm<T, RES> {
private:
  void startTimer(cudaEvent_t &start) {
    cudaEventCreate(&start);
    cudaEventRecord(start);
  }

  float stopTimer(cudaEvent_t &start) {
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
  }

protected:
  const T *mData;
  std::size_t mN;
  bpp::CudaBuffer<T> mCuData;
  bpp::CudaBuffer<RES> mCuResult;

public:
  virtual void initialize(const T *data, std::size_t N, T fromValue,
                          T toValue) override {
    IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue);
    mData = data;
    mN = N;

    CUCH(cudaSetDevice(0));
    mCuData.realloc(N);
    mCuResult.realloc(this->mResult.size());
    mCuResult.memset(0);
  }

  virtual void prepare() override {
    if (!mData || !mN)
      return;
    mCuData.write(mData, mN);
  }

  virtual float run() override {
    if (!this->mData || !this->mN) {
      throw(bpp::RuntimeError() << "No data to process.");
    }
    mCuResult.memset(0);

    cudaEvent_t start;
    startTimer(start);
    run_histogram(*this->mCuData, *this->mCuResult, this->mN, this->mFromValue,
                  this->mToValue);
    return stopTimer(start);
  }

  virtual void finalize() override { mCuResult.read(this->mResult); }
};

#endif

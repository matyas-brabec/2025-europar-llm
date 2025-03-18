#ifndef HISTOGRAM_SERIAL_HPP
#define HISTOGRAM_SERIAL_HPP

#include "interface.hpp"
#include "system/stopwatch.hpp"

#include <cstdint>
#include <vector>

template <typename T = char, typename RES = unsigned int>
class SerialHistogramAlgorithm : public IHistogramAlgorithm<T, RES> {
protected:
  const T *mData;
  std::size_t mN;

public:
  virtual void initialize(const T *data, std::size_t N, T fromValue,
                          T toValue) override {
    IHistogramAlgorithm<T, RES>::initialize(data, N, fromValue, toValue);
    mData = data;
    mN = N;
  }

  virtual float run() override {
    if (!this->mData || !this->mN) {
      throw(bpp::RuntimeError() << "No data to process.");
    }

    for (auto &&r : this->mResult) {
      r = 0;
    }

    bpp::Stopwatch stopwatch(true);

    for (std::size_t i = 0; i < mN; ++i) {
      T val = mData[i];
      if (val >= this->mFromValue && val <= this->mToValue) {
        this->mResult[val - this->mFromValue]++;
      }
    }

    stopwatch.stop();
    return (float)stopwatch.getMiliseconds();
  }
};

#endif

#ifndef HISTOGRAM_INTERFACE_HPP
#define HISTOGRAM_INTERFACE_HPP

#include <cassert>
#include <cstdint>
#include <vector>

void run_histogram(const char *input, unsigned int *histogram,
                   unsigned int inputSize, int from, int to);

/**
 * Interface (abstract class) that specifies how an algorithm has to be
 * implemeted. \tparam T Type of input data elements. \tparam RES Type of output
 * values (histogram counters).
 */
template <typename T = char, typename RES = unsigned int>
class IHistogramAlgorithm {
protected:
  T mFromValue, mToValue;
  std::vector<RES> mResult;

public:
  virtual ~IHistogramAlgorithm() {}

  /**
   * Load input data (e.g., copy them to temporary buffers, perform
   * transformations, ...). This method is not measured by stopwatch.
   */
  virtual void initialize(const T *data, std::size_t N, T fromValue,
                          T toValue) {
    assert(fromValue <= toValue);
    mFromValue = fromValue;
    mToValue = toValue;
    mResult.resize((std::size_t)toValue - (std::size_t)fromValue + 1);
  }

  /**
   * Preparation for execution (e.g., copy data to GPU).
   */
  virtual void prepare() {}

  /**
   * Actual execution of the algorithm.
   * @return float number of miliseconds the algorithm took to run
   */
  virtual float run() = 0;

  /**
   * Finalize (e.g., copy results back from GPU).
   */
  virtual void finalize() {}

  /**
   * Get the result data.
   */
  virtual const std::vector<RES> &getResult() { return mResult; }

  /**
   * Free all internal buffers including the resutls.
   */
  virtual void cleanup() {
    mResult.clear();
    mResult.resize((std::size_t)mToValue - (std::size_t)mFromValue + 1);
  }
};

#endif

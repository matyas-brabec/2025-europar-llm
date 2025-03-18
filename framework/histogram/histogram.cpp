#define _CRT_SECURE_NO_WARNINGS

#include "cuda.hpp"
#include "interface.hpp"
#include "serial.hpp"

#include "cli/args.hpp"
#include "system/file.hpp"
#include "system/mmap_file.hpp"
#include "system/stopwatch.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>

template <typename T = char, typename RES = unsigned int>
std::unique_ptr<IHistogramAlgorithm<T, RES>>
getAlgorithm(const std::string &algoName, bpp::ProgramArguments &args,
             bool quiet = false) {
  using map_t =
      std::map<std::string, std::unique_ptr<IHistogramAlgorithm<T, RES>>>;

  map_t algorithms;
  algorithms["serial"] = std::make_unique<SerialHistogramAlgorithm<T, RES>>();
  algorithms["cuda"] = std::make_unique<CudaHistogramAlgorithm<T, RES>>();

  // PLACE ADDITIONAL ALGORITHMS HERE ...

  auto it = algorithms.find(algoName);
  if (it == algorithms.end()) {
    throw(bpp::RuntimeError() << "Unkown algorithm '" << algoName << "'.");
  }

  if (!quiet) {
    std::cerr << "Selected algorithm: " << algoName << std::endl;
  }
  return std::move(it->second);
}

template <typename T = char, typename RES = unsigned int>
bool verify(const std::vector<RES> &res, const std::vector<RES> &correctRes,
            T fromValue) {
  if (res.size() != correctRes.size()) {
    std::cerr << std::endl
              << "Error: Result size mismatch (" << res.size()
              << " values found, but " << correctRes.size()
              << " values expected)!" << std::endl;
    return false;
  }

  std::size_t errorCount = 0;
  for (std::size_t i = 0; i < res.size(); ++i) {
    if (res[i] != correctRes[i]) {
      if (errorCount == 0)
        std::cerr << std::endl;
      if (++errorCount <= 10) {
        std::cerr << "Error in bucket [" << i << "]: " << res[i]
                  << " != " << correctRes[i] << " (expected)" << std::endl;
      }
    }
  }

  if (errorCount > 0) {
    std::cerr << "Total errors found: " << errorCount << std::endl;
  }

  return errorCount == 0;
}

template <typename T = char, typename RES = unsigned int>
void saveResults(const std::string &fileName, const std::vector<RES> &result,
                 T fromValue) {
  bpp::File file(fileName);
  file.open();
  bpp::TextWriter writer(file, "\n", "\t");

  for (std::size_t i = 0; i < result.size(); ++i) {
    writer.writeToken(i + fromValue);
    writer.writeToken(result[i]);
    writer.writeLine();
  }

  file.close();
}

template <typename F = float>
std::pair<F, F> getMeanAndDeviation(const std::vector<F> &times) {
  if (times.empty()) {
    return std::make_pair((F)0.0, (F)0.0);
  }

  F mean = 0.0f;
  for (auto &&time : times)
    mean += time;
  mean /= times.size();

  F variance = 0.0f;
  for (auto &&time : times)
    variance += (time - mean) * (time - mean);
  variance /= times.size();

  return std::make_pair(mean, std::sqrt(variance));
}

template <typename T = char, typename RES = unsigned int>
int run(bpp::ProgramArguments &args) {
  int res = 0;

  auto algoName = args.getArgString("algorithm").getValue();
  auto algorithm = getAlgorithm<T, RES>(algoName, args);

  std::cerr << "MMaping file '" << args[0] << "' ..." << std::endl;
  bpp::MMapFile file;
  file.open(args[0]);

  auto repeatInput = args.getArgInt("repeatInput").getValue();
  const T *data = (const T *)file.getData();
  std::size_t length = file.length();
  std::vector<T> repeatedData;
  if (repeatInput > 1) {
    std::cerr << "Repeating input file " << repeatInput << "x ..." << std::endl;
    repeatedData.reserve(length * repeatInput);
    while (repeatInput > 0) {
      for (std::size_t i = 0; i < length; ++i)
        repeatedData.push_back(data[i]);
      --repeatInput;
    }

    length = repeatedData.size();
    data = &repeatedData[0];
  } else
    file.populate();

  bpp::Stopwatch stopwatch;

  T fromValue = (T)args.getArgInt("fromValue").getValue();
  T toValue = (T)args.getArgInt("toValue").getValue();
  if (fromValue > toValue)
    std::swap(fromValue, toValue);
  std::cerr << "Initialize (range: " << (int)fromValue << ".." << (int)toValue
            << ", data length " << length << ") ..." << std::endl;
  algorithm->initialize(data, length, fromValue, toValue);

  std::cerr << "Preparations ... ";
  std::cerr.flush();
  stopwatch.start();
  algorithm->prepare();
  stopwatch.stop();
  std::cerr << stopwatch.getMiliseconds() << " ms" << std::endl;

  if (args.getArgBool("warmup").getValue()) {
    std::cerr << "Warmup ... " << std::endl;
    algorithm->run();
  }

  std::size_t iters = args.getArgInt("iterations").getValue();
  std::cerr << "Execution (" << iters << " iterations) ..." << std::endl;

  std::vector<float> times(iters);
  for (auto &&time : times) {
    time = algorithm->run();
  }

  std::cerr << "Finalization ... ";
  std::cerr.flush();
  stopwatch.start();
  algorithm->finalize();
  stopwatch.stop();
  std::cerr << stopwatch.getMiliseconds() << " ms" << std::endl;

  auto result = algorithm->getResult();
  std::string verifResult = "";
  if (args.getArgBool("verify").getValue() && algoName != "serial") {
    std::cerr << "Verifying results ... ";
    std::cerr.flush();
    auto baseAlgorithm = getAlgorithm<T, RES>("serial", args, true);
    baseAlgorithm->initialize(data, length, fromValue, toValue);
    baseAlgorithm->prepare();
    baseAlgorithm->run();
    baseAlgorithm->finalize();

    if (verify(algorithm->getResult(), baseAlgorithm->getResult(),
               (T)args.getArgInt("fromValue").getValue())) {
      verifResult = " OK";
    } else {
      verifResult = " FAILED";
    }

    baseAlgorithm->cleanup();
  }

  auto stats = getMeanAndDeviation(times);
  std::cout << stats.first << " " << stats.second << verifResult << std::endl;

  if (args.getArg("save").isPresent()) {
    auto saveToFile = args.getArgString("save").getValue();
    std::cerr << "Saving results to " << saveToFile << " ..." << std::endl;
    saveResults(saveToFile, algorithm->getResult(),
                (T)args.getArgInt("fromValue").getValue());
  }

  algorithm->cleanup();
  std::cerr << "And we're done here." << std::endl;
  return res;
}

int main(int argc, char *argv[]) {
  /*
   * Arguments
   */
  bpp::ProgramArguments args(1, 1);
  args.setNamelessCaption(0, "Input file");

  try {
    args.registerArg<bpp::ProgramArguments::ArgString>(
        "algorithm", "Which algorithm is to be tested.", false, "serial");
    args.registerArg<bpp::ProgramArguments::ArgString>(
        "save", "Path to a file to which the histogram is saved", false);
    args.registerArg<bpp::ProgramArguments::ArgBool>(
        "verify", "Results will be automatically verified using serial "
                  "algorithm as baseline.");
    args.registerArg<bpp::ProgramArguments::ArgBool>(
        "warmup", "Run the algorithm once before measuring time.");
    args.registerArg<bpp::ProgramArguments::ArgInt>(
        "iterations",
        "Number of iterations to run the algorithm for measurement.", false, 10,
        0, 255);

    args.registerArg<bpp::ProgramArguments::ArgInt>(
        "fromValue", "Ordinal value of the first character in histogram.",
        false, 0, 0, 255);
    args.registerArg<bpp::ProgramArguments::ArgInt>(
        "toValue", "Ordinal value of the last character in histogram.", false,
        127, 0, 255);
    args.registerArg<bpp::ProgramArguments::ArgInt>(
        "repeatInput",
        "Enlarge data input by loading input file multiple times.", false, 1,
        1);

    // Process the arguments ...
    args.process(argc, argv);
  } catch (bpp::ArgumentException &e) {
    std::cerr << "Invalid arguments: " << e.what() << std::endl << std::endl;
    args.printUsage(std::cerr);
    return 1;
  }

  try {
    return run<>(args);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;
    return 2;
  }

  return 0;
}

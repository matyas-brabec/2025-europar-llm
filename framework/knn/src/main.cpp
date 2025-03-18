#include <cmath>
#include <cstddef>
#include <cstdlib>

#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <ratio>

#include <cuda_runtime.h>
#include <omp.h>

#include <cxxopts.hpp>

#include "knn.hpp"

#define CUDA_CHECK(...) \
    do { \
        const cudaError_t status = __VA_ARGS__; \
        if (status != cudaSuccess) { \
            std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(status) << std::endl \
                      << "\t" << #__VA_ARGS__ << std::endl; \
            exit(1); \
        } \
    } while (0)

int main(int argc, char *argv[]) {
    namespace chrono = std::chrono;
    namespace ranges = std::ranges;
    namespace views = std::ranges::views;

    using t_unit = chrono::duration<double, std::milli>; // milliseconds

    // Generate random data
    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // use cxxopts to parse command line arguments
    cxxopts::Options options("knn", "KNN benchmark");
    options.add_options()
        ("n", "number of data points", cxxopts::value<int>())
        ("m", "number of query points", cxxopts::value<int>())
        ("k", "number of neighbors", cxxopts::value<int>())
        ("r", "number of repeats", cxxopts::value<int>())
        ("w", "number of warmup iterations", cxxopts::value<int>()->default_value("3"))
        ("h,help", "Print usage and exit", cxxopts::value<bool>())
        ("s,seed", "Random seed for data generation", cxxopts::value<int>()->default_value("0"))
        ("e,eps", "Epsilon for comparing results", cxxopts::value<float>()->default_value("1e-7"));

    const auto options_result = options.parse(argc, argv);

    if (options_result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (options_result.count("n") == 0 || options_result.count("m") == 0 || options_result.count("k") == 0 || options_result.count("r") == 0) {
        std::cerr << options.help() << std::endl;
        return 1;
    }

    const int n = options_result["n"].as<int>();
    const int k = options_result["k"].as<int>();
    const int m = options_result["m"].as<int>();
    const int repeat = options_result["r"].as<int>();
    const int warmup = options_result["w"].as<int>();
    const int seed = options_result["s"].as<int>();
    const float eps = options_result["e"].as<float>();

    std::vector<t_unit> times;
    times.reserve(repeat);

    std::vector<float2> data(n);
    for (int i = 0; i < n; i++) {
        data[i].x = dis(gen);
        data[i].y = dis(gen);
    }

    std::vector<float2> query(m);
    for (int i = 0; i < m; i++) {
        query[i].x = dis(gen);
        query[i].y = dis(gen);
    }

    // Perform kNN search
    std::vector<std::pair<int, float>> result(m * k);

    float2 *d_query, *d_data;
    std::pair<int, float> *d_result;

    CUDA_CHECK(cudaMalloc(&d_query, m * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_result, m * k * sizeof(std::pair<int, float>)));

    CUDA_CHECK(cudaMemcpy(d_query, query.data(), m * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(float2), cudaMemcpyHostToDevice));

    for (auto r : views::iota(0, warmup)) {
        const auto start = chrono::high_resolution_clock::now();
        run_knn(d_query, m, d_data, n, d_result, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = chrono::high_resolution_clock::now();

        std::cerr << "cuda" << "," << "warmup-run,"
                  << r << "," << n << "," << m << "," << k << "," << seed << ","
                  << chrono::duration_cast<t_unit>(end - start).count() << std::endl;
    }

    for (auto r : views::iota(0, repeat)) {
        const auto start = chrono::high_resolution_clock::now();
        run_knn(d_query, m, d_data, n, d_result, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto end = chrono::high_resolution_clock::now();

        times.push_back(chrono::duration_cast<t_unit>(end - start));

        std::cerr << "cuda" << "," << "run,"
                  << r << "," << n << "," << m << "," << k << "," << seed << ","
                  << chrono::duration_cast<t_unit>(end - start).count() << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(result.data(), d_result, m * k * sizeof(std::pair<int, float>), cudaMemcpyDeviceToHost));

    // Verify the results
    std::vector<std::pair<int, float>> gold_result(m * k);

    const auto gold_start = chrono::high_resolution_clock::now();
    run_knn_baseline(d_query, m, d_data, n, d_result, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto gold_end = chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(gold_result.data(), d_result, m * k * sizeof(std::pair<int, float>), cudaMemcpyDeviceToHost));

    std::cerr << "baseline," << "run,"
              << 0 << "," << n << "," << m << "," << k << "," << seed << ","
              << chrono::duration_cast<t_unit>(gold_end - gold_start).count() << std::endl;

    bool invalid = false;
    for (int i = 0; i < m; i++) {
        // sort the results to ensure the order is the same (hopefully)
        const auto cmp = [eps, k](auto a, auto b) {
            // If the distances are equal, sort by index
            if (a.second == b.second) {
                return a.first < b.first;
            }

            return a.second < b.second;
        };

        const auto start = i * k;
        const auto end = (i + 1) * k;

        std::sort(result.begin() + start, result.begin() + end, cmp);
        std::sort(gold_result.begin() + start, gold_result.begin() + end, cmp);

        bool valid = true;
        for (int j = 0; j < k; j++) {
            const auto [index, distance] = result[i * k + j];
            const auto [gold_index, gold_distance] = gold_result[i * k + j];

            if (index != gold_index) {
                std::cerr << "Mismatch of indices at (query=" << i << ", neighbor_index=" << j << "): got " << index << ", expected " << gold_index << std::endl;
            }
            if (fabs(distance - gold_distance) >= eps) {
                std::cerr << "Mismatch of distances at (query=" << i << ", neighbor_index=" << j << "): got " << distance << ", expected " << gold_distance << std::endl;
                valid = false;
                break;
            }
        }

        if (!valid) {
            invalid = true;
            break;
        }
    }

    using count_t = decltype(std::declval<t_unit>().count());

    struct mean_generator {
        count_t mean;

        count_t operator()(t_unit t) {
            mean += t.count();
            return mean;
        }

        operator count_t() const {
            return mean;
        }
    } mean_generator{0};

    auto mean = (count_t)ranges::for_each(times, mean_generator).fun / times.size();

    struct stddev_generator {
        count_t mean;
        count_t stddev;

        count_t operator()(t_unit t) {
            auto diff = t.count() - mean;
            stddev += diff * diff;
            return stddev;
        }

        operator count_t() const {
            return stddev;
        }
    } stddev_generator{mean, 0};

    auto stddev = sqrt((count_t)ranges::for_each(times, stddev_generator).fun / times.size());

    auto valid = invalid ? "Invalid" : "OK";

    std::cout << mean << ' ' << stddev << ' ' << valid << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_query));
    d_query = nullptr;

    CUDA_CHECK(cudaFree(d_data));
    d_data = nullptr;

    CUDA_CHECK(cudaFree(d_result));
    d_result = nullptr;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

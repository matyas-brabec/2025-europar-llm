#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "gol.cuh"
#include <cuda_runtime.h>

#define seed 42
#define warm_up_iterations 3
#define hot_runs 10
#define check_sum_parts 8
#define BITPACKED_T std::uint64_t
#define WORD_SIZE (sizeof(BITPACKED_T) * 8)

#define LOG std::cerr
#define RESULT std::cout

#define NO_BIT_PACKING 0
#define ROW_PACKING 1
#define TILE_PACKING 2

template <int byte_size>
struct type_of_size {};

template <>
struct type_of_size<1> {
    using type = std::uint8_t;
};

template <>
struct type_of_size<2> {
    using type = std::uint16_t;
};

template <>
struct type_of_size<4> {
    using type = std::uint32_t;
};

template <>
struct type_of_size<8> {
    using type = std::uint64_t;
};

#define BOOL_T type_of_size<sizeof(bool)>::type

#define ASSERT_LAST_CUDA_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit (1); \
        } \
    } while (0)

void perform_gol(const std::vector<BOOL_T>& grid, std::vector<BOOL_T>& out_grid, int dim, int iterations, std::string id);
template <int BIT_PACKING_MODE>
void perform_gol_packed(const std::vector<BOOL_T>& grid, std::vector<BOOL_T>& out_grid, int dim, int iterations, std::string id);
void print_result(std::string id, int iterations, int dim, float milliseconds, const std::vector<BOOL_T>& out_grid);
void parse_args(char* argv[], std::string& id, int& bit_packing_mode, int& dim, int& iterations);
void init_grid(std::vector<BOOL_T>& grid);
std::vector<BITPACKED_T> to_bitpacked_rows(const std::vector<BOOL_T>& grid, int dim);
std::vector<BOOL_T> from_bitpacked_rows(const std::vector<BITPACKED_T>& grid, int dim);
std::vector<BITPACKED_T> to_bitpacked_tiles(const std::vector<BOOL_T>& grid, int dim);
std::vector<BOOL_T> from_bitpacked_tiles(const std::vector<BITPACKED_T>& grid, int dim);
template<typename T>
void init_gpu_mem(const std::vector<T>& grid, T** d_grid, T** d_new_grid);
template <typename T>
void free_gpu_mem(T* d_grid, T* d_new_grid);
std::string get_check_sums(const std::vector<BOOL_T>& grid);
void start_cuda_timer(cudaEvent_t& start);
float stop_cuda_timer(cudaEvent_t& start);
void print_grid(const std::vector<BOOL_T>& grid, int dim);

int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <string id> <bit packing mode> <grid size> <iterations>\n";
        return 1;
    }

    std::string id;
    int dim, iterations, bit_packing_mode;

    parse_args(argv, id, bit_packing_mode, dim, iterations);

    LOG << "Initializing grid" << std::endl;

    std::vector<BOOL_T> grid(dim * dim, 0);
    std::vector<BOOL_T> out_grid(dim * dim, 0);

    init_grid(grid);

    LOG << "Running" << std::endl;

    RESULT << "id;iterations;dim;time;checksum;seed" << std::endl;

    for (int i = 0; i < warm_up_iterations + hot_runs; ++i) {
        LOG << " -- Iteration " << i << std::endl;

        if (bit_packing_mode == NO_BIT_PACKING) {
            perform_gol(grid, out_grid, dim, iterations, id);
        } else if (bit_packing_mode == ROW_PACKING) {
            perform_gol_packed<ROW_PACKING>(grid, out_grid, dim, iterations, id);
        } else if (bit_packing_mode == TILE_PACKING) {
            perform_gol_packed<TILE_PACKING>(grid, out_grid, dim, iterations, id);
        } else {
            std::cerr << "Invalid bit packing mode" << std::endl;
            return 1;
        }
    }
}

void perform_gol(const std::vector<BOOL_T>& grid, std::vector<BOOL_T>& out_grid, int dim, int iterations, std::string id) {
    BOOL_T* d_grid;
    BOOL_T* d_new_grid;

    init_gpu_mem(grid, &d_grid, &d_new_grid);

    initialize_internal_data_structures(dim);

    cudaEvent_t start;
    start_cuda_timer(start);

    bool* d_grid_bool = reinterpret_cast<bool*>(d_grid);
    bool* d_new_grid_bool = reinterpret_cast<bool*>(d_new_grid);

    for (int i = 0; i < iterations; ++i) {
        run_game_of_life(d_grid_bool, d_new_grid_bool, dim);
        ASSERT_LAST_CUDA_ERROR();

        std::swap(d_grid_bool, d_new_grid_bool);
    }

    float milliseconds = stop_cuda_timer(start);

    cudaMemcpy(out_grid.data(), d_grid, out_grid.size() * sizeof(BOOL_T), cudaMemcpyDeviceToHost);
    free_gpu_mem(d_grid, d_new_grid);

    print_result(id, iterations, dim, milliseconds, out_grid);
}

template <int BIT_PACKING_MODE>
void perform_gol_packed(const std::vector<BOOL_T>& grid, std::vector<BOOL_T>& out_grid, int dim, int iterations, std::string id) {
    std::vector<BITPACKED_T> packed_grid;
    std::vector<BITPACKED_T> packed_out_grid(dim * dim / WORD_SIZE, 0);

    auto tiles = to_bitpacked_tiles(grid, dim);
    auto from = from_bitpacked_tiles(tiles, dim);

    if (BIT_PACKING_MODE == ROW_PACKING) {
        packed_grid = to_bitpacked_rows(grid, dim);
    } else if (BIT_PACKING_MODE == TILE_PACKING) {
        packed_grid = to_bitpacked_tiles(grid, dim);
    } else {
        std::cerr << "Invalid bit packing mode" << std::endl;
        return;
    }

    BITPACKED_T* d_grid;
    BITPACKED_T* d_new_grid;

    init_gpu_mem(packed_grid, &d_grid, &d_new_grid);

    initialize_internal_data_structures(dim);

    cudaEvent_t start;
    start_cuda_timer(start);

    for (int i = 0; i < iterations; ++i) {
        run_game_of_life(d_grid, d_new_grid, dim);
        ASSERT_LAST_CUDA_ERROR();

        std::swap(d_grid, d_new_grid);
    }

    float milliseconds = stop_cuda_timer(start);

    cudaMemcpy(packed_out_grid.data(), d_grid, packed_out_grid.size() * sizeof(BITPACKED_T), cudaMemcpyDeviceToHost);
    free_gpu_mem(d_grid, d_new_grid);

    if (BIT_PACKING_MODE == ROW_PACKING) {
        out_grid = from_bitpacked_rows(packed_out_grid, dim);
    } else if (BIT_PACKING_MODE == TILE_PACKING) {
        out_grid = from_bitpacked_tiles(packed_out_grid, dim);
    }

    print_result(id, iterations, dim, milliseconds, out_grid);
}

void print_result(std::string id, int iterations, int dim, float milliseconds, const std::vector<BOOL_T>& out_grid) {
    RESULT << id << ";" << iterations << ";" << dim << ";" << milliseconds << ";" << get_check_sums(out_grid) << ";" << seed << std::endl;
}

void parse_args(char* argv[], std::string& id, int& bit_packing_mode, int& dim, int& iterations) {
    id = argv[1];
    bit_packing_mode = std::stoi(argv[2]);
    dim = std::stoi(argv[3]);
    iterations = std::stoi(argv[4]);
}

void init_grid(std::vector<BOOL_T>& grid) {
    std::srand(seed);
    for (auto& cell : grid) {
        cell = std::rand() % 2;
    }
}

std::vector<BITPACKED_T> to_bitpacked_rows(const std::vector<BOOL_T>& grid, int dim) {
    std::vector<BITPACKED_T> new_grid(dim * dim / WORD_SIZE, 0);

    for (std::size_t i = 0; i < grid.size(); i += WORD_SIZE) {
        BITPACKED_T row = 0;

        for (std::size_t j = 0; j < WORD_SIZE; ++j) {
            if (grid[i + j]) {
                row |= 1ULL << j;
            }
        }

        new_grid[i / WORD_SIZE] = row;
    }

    return new_grid;
}

#define X_bits 8
#define Y_bits 8

std::vector<BITPACKED_T> to_bitpacked_tiles(const std::vector<BOOL_T>& grid, int dim) {
    std::vector<BITPACKED_T> new_grid(dim * dim / WORD_SIZE, 0);
    auto raw_data = grid.data();

    for (int y = 0; y < dim; y += Y_bits) {
        for (int x = 0; x < dim; x += X_bits) {

            BITPACKED_T word = 0;
            auto bit_setter = static_cast<BITPACKED_T>(1) << (sizeof(BITPACKED_T) * 8 - 1);

            for (int y_bit = 0; y_bit < Y_bits; ++y_bit) {
                for (int x_bit = 0; x_bit < X_bits; ++x_bit) {

                    BITPACKED_T value = raw_data[(y + y_bit) * dim + (x + x_bit)] ? 1 : 0;

                    if (value) {
                        word |= bit_setter;
                    }

                    bit_setter = bit_setter >> 1;
                }
            }

            new_grid[(y / Y_bits) * (dim / X_bits) + x / X_bits] = word;
        }
    }

    return new_grid;
}

std::vector<BOOL_T> from_bitpacked_rows(const std::vector<BITPACKED_T>& grid, int dim) {
    std::vector<BOOL_T> new_grid(dim * dim, 0);

    for (std::size_t word_idx = 0; word_idx < grid.size(); ++word_idx) {
        for (std::size_t bit = 0; bit < WORD_SIZE; ++bit) {
            new_grid[word_idx * WORD_SIZE + bit] = (grid[word_idx] >> bit) & 1;
        }
    }

    return new_grid;
}

std::vector<BOOL_T> from_bitpacked_tiles(const std::vector<BITPACKED_T>& grid, int dim) {
    std::vector<BOOL_T> new_grid(dim * dim, 0);
    auto raw_data = new_grid.data();

    for (int y = 0; y < dim; y += Y_bits) {
        for (int x = 0; x < dim; x += X_bits) {

            auto word = grid[(y / Y_bits) * (dim / X_bits) + x / X_bits];
            auto mask = static_cast<BITPACKED_T>(1) << (sizeof(BITPACKED_T) * 8 - 1);

            for (int y_bit = 0; y_bit < Y_bits; ++y_bit) {
                for (int x_bit = 0; x_bit < X_bits; ++x_bit) {

                    auto value = (word & mask) ? 1 : 0;

                    raw_data[(y + y_bit) * dim + (x + x_bit)] = value;

                    mask = mask >> 1;
                }
            }
        }
    }

    return new_grid;
}

template<typename T>
void init_gpu_mem(const std::vector<T>& grid, T** d_grid, T** d_new_grid) {
    cudaMalloc(d_grid, grid.size() * sizeof(T));
    cudaMalloc(d_new_grid, grid.size() * sizeof(T));

    cudaMemcpy(*d_grid, grid.data(), grid.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void free_gpu_mem(T* d_grid, T* d_new_grid) {
    cudaFree(d_grid);
    cudaFree(d_new_grid);
}

std::string get_check_sums(const std::vector<BOOL_T>& grid) {
    std::string output = "";

    for (std::size_t i = 0; i < check_sum_parts; ++i) {
        std::size_t check_sum = 0;

        for (std::size_t j = 0; j < grid.size() / check_sum_parts; ++j) {
            check_sum += grid[i * grid.size() / check_sum_parts + j];
        }

        if (i == 0) {
            output += std::to_string(check_sum);
        } else {
            output += "-" + std::to_string(check_sum);
        }
    }

    return output;
}

void start_cuda_timer(cudaEvent_t& start) {
    cudaEventCreate(&start);
    cudaEventRecord(start);
}

float stop_cuda_timer(cudaEvent_t& start) {
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

void print_grid(const std::vector<BOOL_T>& grid, int dim) {
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            std::cout << (grid[y * dim + x] ? '#' : '.');
        }
        std::cout << '\n';
    }
    std::cout << "-----------------" << std::endl;
}

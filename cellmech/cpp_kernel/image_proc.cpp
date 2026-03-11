#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <cmath>
#include <limits>

namespace nb = nanobind;

// C++ Implementation of K-Means
nb::ndarray<nb::numpy, int> k_means_cpp(
    nb::ndarray<nb::numpy, float, nb::shape<nb::any, nb::any>> data, 
    int k, 
    int max_iters) 
{
    size_t n_points = data.shape(0);
    size_t n_dims = data.shape(1);

    // 1. Initialize centroids (simplified: first K points)
    std::vector<std::vector<float>> centroids(k, std::vector<float>(n_dims));
    for (int i = 0; i < k; ++i) {
        for (size_t j = 0; j < n_dims; ++j) {
            centroids[i][j] = data(i, j);
        }
    }

    // Prepare labels array to return
    int* labels_ptr = new int[n_points];

    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment Step
        for (size_t i = 0; i < n_points; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            for (int c = 0; c < k; ++c) {
                float dist = 0;
                for (size_t d = 0; d < n_dims; ++d) {
                    float diff = data(i, d) - centroids[c][d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            labels_ptr[i] = best_cluster;
        }
        // (Update step omitted for brevity, usually involves averaging points)
    }

    // Return as a NumPy-compatible array
    size_t shape[1] = { n_points };
    return nb::ndarray<nb::numpy, int>(labels_ptr, 1, shape, nb::handle());
}

NB_MODULE(my_kmeans, m) {
    m.def("k_means", &k_means_cpp, 
          nb::arg("data"), nb::arg("k"), nb::arg("max_iters") = 100);
}
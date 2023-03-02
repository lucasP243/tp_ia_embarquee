#include <inttypes.h>
#include <stdio.h>

#include "mnist_data_handler.h"
#include "neural_network.h"

#define SAMPLING_SIZE 600

size_t get_max_index(const float *array, const size_t array_size)
{
    size_t max_index = 0;
    float max_value = array[0];

    for (size_t i = 0; i < array_size; i++)
    {
        if (array[i] > max_value)
        {
            max_index = i;
            max_value = array[i];
        }
    }

    return max_index;
}

int main(int argc, char *argv[])
{
    if (mnist_data_handler_init() != 0) return -1;

    struct neural_network nn;

    init_random_weights(&nn);

    // Training
    neural_network_train(&nn, SAMPLING_SIZE);

    // Testing
    uint8_t image_buffer[IMAGE_SIZE];
    uint8_t desired_output;

    uint32_t error_count;

    for (size_t i = 0; i < TESTING_SET_LENGTH; i++)
    {
        if (mnist_get_testing_sample(i, image_buffer, &desired_output) != 0)
        {
            return -1;
        }

        load_input(&nn, image_buffer, desired_output);

        (void) neural_network_compute(&nn);

        uint8_t result = (uint8_t) get_max_index(nn.output_layer, OUTPUT_LAYER_COUNT);
        if (result != desired_output)
        {
            printf("Prediction error on sample %" PRId64 " (expected %" PRId8 ", predicted %" PRId8 ")\n", i, desired_output, result);
            error_count += 1;
        }
    }

    float accuracy = 100.0f * (1.0f - (float) error_count / (float) TESTING_SET_LENGTH);
    printf("Testing complete. %" PRId32 " errors out of %" PRId32 " samples (%.2f%% accuracy).\n", error_count, TESTING_SET_LENGTH, accuracy);

    
    if (mnist_data_handler_close() != 0) return -1;

    return 0;
}
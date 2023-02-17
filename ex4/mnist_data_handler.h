#ifndef _MNIST_DATA_HANDLER_H_
#define _MNIST_DATA_HANDLER_H_

#include <stdint.h>
#include <stddef.h>

#define TRAINING_SET_LENGTH 60000
#define TESTING_SET_LENGTH  10000

#define IMAGE_HEIGHT        28
#define IMAGE_WIDTH         28
#define IMAGE_SIZE          (IMAGE_HEIGHT * IMAGE_WIDTH)

int mnist_data_handler_init();

int mnist_data_handler_close();

int mnist_get_training_sample(size_t index, uint8_t image[IMAGE_SIZE], uint8_t *label);

int mnist_get_testing_sample(size_t index, uint8_t image[IMAGE_SIZE], uint8_t *label);

#endif // _MNIST_DATA_HANDLER_H_
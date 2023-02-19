#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdint.h>

#include "mnist_data_handler.h"

#define INPUT_LAYER_COUNT	(IMAGE_SIZE)
#define HIDDEN_LAYER_COUNT	100
#define OUTPUT_LAYER_COUNT	10

#define INPUT_RANGE		(255.0f)

#define LEARNING_RATE	(0.01f)

struct neural_network
{
	float	input_layer		[INPUT_LAYER_COUNT];

	float	weights1		[INPUT_LAYER_COUNT][HIDDEN_LAYER_COUNT];

	float 	hidden_layer	[HIDDEN_LAYER_COUNT];

	float	weights2		[HIDDEN_LAYER_COUNT][OUTPUT_LAYER_COUNT];

	float 	output_layer	[OUTPUT_LAYER_COUNT];

	uint8_t desired_output	[OUTPUT_LAYER_COUNT];
};

void	load_input(struct neural_network *, const uint8_t input_layer[IMAGE_SIZE], const uint8_t desired_output);

void	init_random_weights(struct neural_network *);

int		neural_network_train(struct neural_network *, const size_t sample_size);

float	neural_network_compute(struct neural_network *);

#endif // NEURAL_NETWORK_H

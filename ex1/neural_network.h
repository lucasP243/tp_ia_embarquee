#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>
#include <stdint.h>

#define IMAGE_COUNT		(2)
#define IMAGE_HEIGHT	(8)
#define IMAGE_WIDTH		(6)

#define INPUT_COUNT		(IMAGE_HEIGHT * IMAGE_WIDTH)
#define OUTPUT_COUNT	(1)

#define THRESHOLD		(0.5f)
#define LEARNING_RATE	(0.01f)

inline float random_f(float lower_bound, float upper_bound)
{
    // Gives an random floating point number between 0 and 1
    float random = (float) rand() / (float) RAND_MAX;

    return lower_bound + random * (upper_bound - lower_bound);
}

struct neural_network {
	uint8_t input_layer		[INPUT_COUNT];
	uint8_t output_layer	[OUTPUT_COUNT];

	uint8_t desired_output;

	float	weights			[INPUT_COUNT][OUTPUT_COUNT];
};

int		load_image_input(struct neural_network *, const size_t image_index, const float noise);

void	init_random_weights(struct neural_network *);

int		neural_network_compute(struct neural_network *);

int		neural_network_train(struct neural_network *);

#endif // NEURAL_NETWORK_H

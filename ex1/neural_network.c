#include "neural_network.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>

// private

static const char *learning_plot_filepath = "./learning_curve.dat";

static const char *IMAGE_LIST[] = {
		"motifs/zero.txt",	// [0]
		"motifs/un.txt",	// [1]
};

extern inline float random_f(float, float);

float run_neural_network(struct neural_network *nn, bool is_learning)
{
	float error = 0;

	for (size_t output_index = 0; output_index < OUTPUT_COUNT; output_index++)
	{
		float potential = 0.0f;

		for (size_t input_index = 0; input_index < INPUT_COUNT; input_index++)
		{
			potential += nn->input_layer[input_index] * nn->weights[input_index][output_index];
		}

		nn->output_layer[output_index] = (potential < THRESHOLD) ? 0 : 1;

		float output_error = nn->desired_output - nn->output_layer[output_index];

		error += output_error;

		if (is_learning == true)
		{
			for (size_t input_index = 0; input_index < INPUT_COUNT; input_index++)
			{
				nn->weights[input_index][output_index] += LEARNING_RATE * (float) (output_error * nn->input_layer[input_index]);
			}
		}
	}

	return error;
}

// public

int load_image_input(struct neural_network *nn, size_t image_index, float noise)
{
	FILE *image_file = fopen(IMAGE_LIST[image_index], "r");

	if (image_file == NULL)
	{
		perror("Failed to open image file");
		return -1;
	}

	for (size_t i = 0; i < IMAGE_HEIGHT; i++)
	{
		for (size_t j = 0; j < IMAGE_WIDTH; j++)
		{
			size_t input_index = i * IMAGE_WIDTH + j;
			char image_pixel;
			size_t bytes_read;

			if ((bytes_read = fread(&image_pixel, sizeof image_pixel, 1, image_file)) != 1)
    		{
    			fprintf(stderr, "Failed to read byte (read %" PRId64 " bytes instead).\n", bytes_read);
    			return -1;
    		}

			nn->input_layer[input_index] = (image_pixel == '*') ? 1 : 0;

			if (noise > 0.0f && random_f(0.0f, 1.0f) < noise)
			{
				nn->input_layer[input_index] = abs(nn->input_layer[input_index] - 1);
			}
		}

		if(fseek(image_file, 1, SEEK_CUR) == -1)
		{
			perror("Failed to skip carriage return");
			return -1;
		}
	}

	if (fscanf(image_file, "%"SCNd8, &nn->desired_output) != 1)
	{
		perror("Failed to read desired output");
		return -1;
	}

	if (fclose(image_file) == EOF)
	{
		perror("Failed to close file");
		return -1;
	}

	return 0;
}

void init_random_weights(struct neural_network *nn)
{
	srand(time(NULL));

	for (size_t i = 0; i < INPUT_COUNT; i++)
	{
		for (size_t j = 0; j < OUTPUT_COUNT; j++)
		{
			// Generate random weight in [-1/48; 1/48]
			nn->weights[i][j] = random_f(-1.0f, 1.0f) / (float) INPUT_COUNT;
		}
	}
}

int neural_network_compute(struct neural_network *nn)
{
	(void) run_neural_network(nn, false);

	return nn->output_layer[0];
}

int neural_network_train(struct neural_network *nn)
{
	const float	noise_level	= 0.0f;
	const bool	is_learning	= true;

	uint8_t total_error = 1;

	FILE *learning_plot_file = fopen(learning_plot_filepath, "w");
	if (learning_plot_file == NULL)
	{
		perror("Failed to open file");
		return -1;
	}

	printf("========== BEGIN LEARNING ==========\n");

	for (uint32_t iteration = 0; total_error > 0; iteration++)
	{
		total_error = 0.0f;

		for (size_t image_index = 0; image_index < IMAGE_COUNT; image_index++)
		{
			// TODO handle error
			(void) load_image_input(nn, image_index, noise_level);

			const float error = run_neural_network(nn, is_learning);

			total_error += abs(error);
		}

		printf("Iteration %"PRId32"\ttotal error = %"PRId8"\n", iteration, total_error);
		fprintf(learning_plot_file, "%"PRId32"\t%"PRId8"\n", iteration, total_error);
	}

	printf("======== FINISHED LEARNING =========\n");

	if (fclose(learning_plot_file) == EOF)
	{
		perror("Failed to close file");
		return -1;
	}

	return 0;
}

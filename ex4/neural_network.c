#include "neural_network.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <string.h>

// private

static const char *learning_plot_filepath = "./learning_curve.dat";

float random_f(float lower_bound, float upper_bound)
{
    // Gives an random floating point number between 0 and 1
    float random = (float) rand() / (float) RAND_MAX;

    return lower_bound + random * (upper_bound - lower_bound);
}

float activation(float x)
{
	return 1 / (1 + expf(-x));
}

float activation_derived(float x)
{
	float f_x = activation(x);
	return f_x * (1 - f_x);
}

float run_neural_network(struct neural_network *nn, bool is_learning)
{
	float pot_h		[HIDDEN_LAYER_COUNT] = {0.0f};
	float pot_j		[OUTPUT_LAYER_COUNT] = {0.0f};
	float error_h	[HIDDEN_LAYER_COUNT] = {0.0f};
	float error_j	[OUTPUT_LAYER_COUNT] = {0.0f};

	for (size_t h = 0; h < HIDDEN_LAYER_COUNT; h++)
	{
		pot_h[h] = 0;

		for (size_t i = 0; i < INPUT_LAYER_COUNT; i++)
		{
			pot_h[h] += nn->input_layer[i] * nn->weights1[i][h];
		}

		nn->hidden_layer[h] = activation(pot_h[h]);
	}

	for (size_t j = 0; j < OUTPUT_LAYER_COUNT; j++)
	{
		pot_j[j] = 0;

		for (size_t h = 0; h < HIDDEN_LAYER_COUNT; h++)
		{
			pot_j[j] += nn->hidden_layer[h] * nn->weights2[h][j];
		}

		nn->output_layer[j] = activation(pot_j[j]);

		error_j[j] = activation_derived(pot_j[j]) * (nn->output_layer[j] - nn->desired_output[j] );
	}

	if (is_learning)
	{
		// retropropagate error to hidden layer
		for (size_t h = 0; h < HIDDEN_LAYER_COUNT; h++)
		{
			float retro_error = 0;
			for (size_t j = 0; j < OUTPUT_LAYER_COUNT; j++)
			{
				retro_error += error_j[j] * nn->weights2[h][j];
			}
			error_h[h] = activation_derived(pot_h[h]) * retro_error;

			// learn
			for (size_t i = 0; i < INPUT_LAYER_COUNT; i++)
			{
				nn->weights1[i][h] -= LEARNING_RATE * error_h[h] * nn->input_layer[i];
			}
			for (size_t j = 0; j < OUTPUT_LAYER_COUNT; j++)
			{
				nn->weights2[h][j] -= LEARNING_RATE * error_j[j] * nn->hidden_layer[h];
			}
		}
	}

	float abs_error = 0;
	for (size_t j = 0; j < OUTPUT_LAYER_COUNT; j++)
	{
		abs_error += fabs(error_j[j]);
	}
	return abs_error;
}

// public

void load_input(struct neural_network *nn, const uint8_t input_layer[IMAGE_SIZE], const uint8_t desired_output)
{
	for (size_t i = 0; i < INPUT_LAYER_COUNT; i++)
	{
		nn->input_layer[i] = (float) input_layer[i] / INPUT_RANGE;
	}

	(void) memset(&nn->hidden_layer, 0, HIDDEN_LAYER_COUNT * sizeof &nn->hidden_layer);

	(void) memset(&nn->desired_output, 0, OUTPUT_LAYER_COUNT);
	nn->desired_output[desired_output] = 1;
}

void init_random_weights(struct neural_network *nn)
{
	srand(time(NULL));

	for (size_t h = 0; h < HIDDEN_LAYER_COUNT; h++)
	{
		for (size_t i = 0; i < INPUT_LAYER_COUNT; i++)
		{
			// Generate random weight in [-1/784; 1/784]
			nn->weights1[i][h] = random_f(-1.0f, 1.0f) / (float) INPUT_LAYER_COUNT;
		}

		for (size_t j = 0; j < OUTPUT_LAYER_COUNT; j++)
		{
			// Generate random weight in [-1/100; 1/100]
			nn->weights2[h][j] = random_f(-1.0f, 1.0f) / (float) HIDDEN_LAYER_COUNT;
		}
	}
}

float neural_network_compute(struct neural_network *nn)
{
	return run_neural_network(nn, false);
}

int neural_network_train(struct neural_network *nn, const size_t sample_size)
{
	const bool is_learning = true;

	uint8_t image_buffer[IMAGE_SIZE];
	uint8_t desired_output;

	float total_error = 1.0f;

	FILE *learning_plot_file = fopen(learning_plot_filepath, "w");
	if (learning_plot_file == NULL)
	{
		perror("Failed to open file");
		return -1;
	}

	printf("========== BEGIN LEARNING ==========\n");

	for (uint32_t iteration = 0; total_error > 0.04f; iteration++)
	{
		total_error = 0.0f;

		for (size_t _ = 0; _ < sample_size; _++)
		{
			size_t sample_index = rand() % TRAINING_SET_LENGTH;

			if (mnist_get_training_sample(sample_index, image_buffer, &desired_output) != 0)
			{
				fprintf(stderr, "Training aborted due to error.\n");
				return -1;
			}

			load_input(nn, image_buffer, desired_output);

			const float error = run_neural_network(nn, is_learning);

			total_error += error;
		}

		total_error /= (float) sample_size;

		printf("Iteration %"PRId32"\ttotal error = %.3f\n", iteration, total_error);
		fprintf(learning_plot_file, "%"PRId32"\t%.3f\n", iteration, total_error);
	}

	printf("======== FINISHED LEARNING =========\n");

	if (fclose(learning_plot_file) == EOF)
	{
		perror("Failed to close file");
		return -1;
	}

	return 0;
}

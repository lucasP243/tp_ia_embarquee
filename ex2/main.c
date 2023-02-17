#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

#include "neural_network.h"

#define RESULT_DATA_0_FILEPATH "efficiency_curve_0.dat"
#define RESULT_DATA_1_FILEPATH "efficiency_curve_1.dat"

#define NOISE_MIN	(0.02f)
#define NOISE_MAX	(0.50f)
#define NOISE_STEP	(0.01f)

#define NB_ITER (50)

int main(int argc, char *argv[])
{
	struct neural_network nn;

	init_random_weights(&nn);
	neural_network_train(&nn);

	// Neural network is trained, testing with noise

	FILE *result_file_0 = fopen(RESULT_DATA_0_FILEPATH, "w");
	if (result_file_0 == NULL)
	{
		perror("Opening file " RESULT_DATA_0_FILEPATH);
		return EXIT_FAILURE;
	}

	FILE *result_file_1 = fopen(RESULT_DATA_1_FILEPATH, "w");
	if (result_file_1 == NULL)
	{
		perror("Opening file " RESULT_DATA_1_FILEPATH);
		return EXIT_FAILURE;
	}

	for (float noise = NOISE_MIN; noise < NOISE_MAX; noise += NOISE_STEP)
	{
		uint8_t success_count_0 = 0;
		uint8_t success_count_1 = 0;

		for (uint8_t iteration = 0; iteration < NB_ITER; iteration++)
		{
			load_image_input(&nn, 0, noise);
			if (neural_network_compute(&nn) == nn.desired_output)
			{
				success_count_0 += 1;
			}

			load_image_input(&nn, 1, noise);
			if (neural_network_compute(&nn) == nn.desired_output)
			{
				success_count_1 += 1;
			}
		}

		float success_rate_0 = 100.0 * (float) success_count_0 / (float) NB_ITER;
		float success_rate_1 = 100.0 * (float) success_count_1 / (float) NB_ITER;

		(void) fprintf(result_file_0, "%" PRId8 " %.2f\n", (uint8_t) (100 * noise), success_rate_0);
		(void) fprintf(result_file_1, "%" PRId8 " %.2f\n", (uint8_t) (100 * noise), success_rate_1);
	}

	if (fclose(result_file_0) == EOF)
	{
		perror("Closing file " RESULT_DATA_0_FILEPATH);
		return EXIT_FAILURE;
	}

	if (fclose(result_file_1) == EOF)
	{
		perror("Closing file " RESULT_DATA_1_FILEPATH);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

#include <inttypes.h>
#include <stdio.h>

#include "mnist_data_handler.h"

int main(int argc, char *argv[])
{
    if (mnist_data_handler_init() != 0) return -1;

    uint8_t image[IMAGE_SIZE];
    uint8_t label;

    if (mnist_get_training_sample(35354, image, &label) != 0) return -1;

    for(int i = 0; i < IMAGE_SIZE; i++)
    {
        printf("%03" PRId8, image[i]);
        if ((i + 1) % IMAGE_WIDTH == 0) printf("\n");
    }

    printf("%" PRId8 "\n", label);
    
    if (mnist_data_handler_close() != 0) return -1;

    return 0;
}
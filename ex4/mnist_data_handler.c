#include "mnist_data_handler.h"

#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

/*****************************************************************************/

#define T10K_IMAGES_FILEPATH        "mnist_data/t10k-images-idx3-ubyte"
#define T10K_IMAGES_HEADER_HI64     (0x0000080300002710)
#define T10K_IMAGES_HEADER_LO64     (0x0000001C0000001C)
#define T10K_IMAGES_COUNT           10000
#define T10K_IMAGES_ROWS            28
#define T10K_IMAGES_COLUMNS         28
#define T10K_IMAGES_SIZE            (T10K_IMAGES_ROWS * T10K_IMAGES_COLUMNS)

static FILE *t10k_images_file = NULL;

/*****************************************************************************/

#define T10K_LABELS_FILEPATH        "mnist_data/t10k-labels-idx1-ubyte"
#define T10K_LABELS_HEADER          (0x0000080100002710)
#define T10K_LABELS_COUNT           10000

static FILE *t10k_labels_file = NULL;

/*****************************************************************************/

#define TRAIN_IMAGES_FILEPATH       "mnist_data/train-images-idx3-ubyte"
#define TRAIN_IMAGES_HEADER_HI64    (0x000008030000EA60)
#define TRAIN_IMAGES_HEADER_LO64    (0x0000001C0000001C)
#define TRAIN_IMAGES_COUNT          60000
#define TRAIN_IMAGES_ROWS           28
#define TRAIN_IMAGES_COLUMNS        28
#define TRAIN_IMAGES_SIZE           (TRAIN_IMAGES_ROWS * TRAIN_IMAGES_COLUMNS)

static FILE *train_images_file = NULL;

/*****************************************************************************/

#define TRAIN_LABELS_FILEPATH       "mnist_data/train-labels-idx1-ubyte"
#define TRAIN_LABELS_HEADER         (0x000008010000EA60)
#define TRAIN_LABELS_COUNT          60000

static FILE *train_labels_file = NULL;

/*****************************************************************************/

/** Shorthand to iterate over files */
#define MNIST_FILE_COUNT 4
struct { FILE **fd; char *filepath; } mnist_file[MNIST_FILE_COUNT] = {
    { .fd = &t10k_images_file, .filepath = T10K_IMAGES_FILEPATH },
    { .fd = &t10k_labels_file, .filepath = T10K_LABELS_FILEPATH },
    { .fd = &train_images_file, .filepath = TRAIN_IMAGES_FILEPATH },
    { .fd = &train_labels_file, .filepath = TRAIN_LABELS_FILEPATH },
};

int mnist_data_handler_init()
{
    errno = 0;

    #pragma unroll
    for (size_t i = 0; i < MNIST_FILE_COUNT; i++)
    {
        if (*mnist_file[i].fd == NULL && (*mnist_file[i].fd = fopen(mnist_file[i].filepath, "r")) == NULL)
        {
            fprintf(stderr, "[ERROR] Failed to open file %s: %s\n", mnist_file[i].filepath, strerror(errno));
        }
    }

    return errno;
}

int mnist_data_handler_close()
{
    errno = 0;

    #pragma unroll
    for (size_t i = 0; i < MNIST_FILE_COUNT; i++)
    {
        if (*mnist_file[i].fd == NULL && fclose(*mnist_file[i].fd) == EOF)
        {
            fprintf(stderr, "[ERROR] Failed to close file %s: %s\n", mnist_file[i].filepath, strerror(errno));
        }

        *mnist_file[i].fd = NULL;
    }

    return errno;
}
#include "mnist_data_handler.h"

#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

/*****************************************************************************/

#define T10K_IMAGES_FILEPATH        "mnist_data/t10k-images-idx3-ubyte"
#define T10K_IMAGES_HEADER_HI64     (0x0000080300002710)
#define T10K_IMAGES_HEADER_LO64     (0x0000001C0000001C)
#define T10K_IMAGES_HEADER_SIZE     16

static FILE *t10k_images_file = NULL;

/*****************************************************************************/

#define T10K_LABELS_FILEPATH        "mnist_data/t10k-labels-idx1-ubyte"
#define T10K_LABELS_HEADER          (0x0000080100002710)
#define T10K_LABELS_HEADER_SIZE     8

static FILE *t10k_labels_file = NULL;

/*****************************************************************************/

#define TRAIN_IMAGES_FILEPATH       "mnist_data/train-images-idx3-ubyte"
#define TRAIN_IMAGES_HEADER_HI64    (0x000008030000EA60)
#define TRAIN_IMAGES_HEADER_LO64    (0x0000001C0000001C)
#define TRAIN_IMAGES_HEADER_SIZE    16

static FILE *train_images_file = NULL;

/*****************************************************************************/

#define TRAIN_LABELS_FILEPATH       "mnist_data/train-labels-idx1-ubyte"
#define TRAIN_LABELS_HEADER         (0x000008010000EA60)
#define TRAIN_LABELS_HEADER_SIZE    8

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
    int error = 0;

    for (size_t i = 0; i < MNIST_FILE_COUNT; i++)
    {
        if (*mnist_file[i].fd == NULL && (*mnist_file[i].fd = fopen(mnist_file[i].filepath, "r")) == NULL)
        {
            error = -1;
            fprintf(stderr, "[ERROR] Failed to open file %s: %s\n", mnist_file[i].filepath, strerror(errno));
        }
    }

    return error;
}

int mnist_data_handler_close()
{
    int error = 0;

    for (size_t i = 0; i < MNIST_FILE_COUNT; i++)
    {
        if (*mnist_file[i].fd == NULL && fclose(*mnist_file[i].fd) == EOF)
        {
            error = -1;
            fprintf(stderr, "[ERROR] Failed to close file %s: %s\n", mnist_file[i].filepath, strerror(errno));
        }

        *mnist_file[i].fd = NULL;
    }

    return error;
}

int mnist_get_training_sample(size_t index, uint8_t image[IMAGE_SIZE], uint8_t *label)
{
    if (index >= TRAINING_SET_LENGTH)
    {
        errno = EINVAL;
        return -1;
    }

    size_t offset_image = TRAIN_IMAGES_HEADER_SIZE + (index * IMAGE_SIZE);
    size_t offset_label = TRAIN_LABELS_HEADER_SIZE + index;

    // No need to worry about conversion from size_t (ulong) to long,
    // because offset will never exceed 47'039'232
    if (fseek(train_images_file, offset_image, SEEK_SET) != 0)
    {
        fprintf(stderr, "[ERROR] Failed to navigate " TRAIN_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fread(image, 1, IMAGE_SIZE, train_images_file) != IMAGE_SIZE)
    {
        fprintf(stderr, "[ERROR] Failed to read from " TRAIN_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fseek(train_labels_file, offset_label, SEEK_SET) != 0)
    {
        fprintf(stderr, "[ERROR] Failed to navigate " TRAIN_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fread(label, 1, 1, train_labels_file) != 1)
    {
        fprintf(stderr, "[ERROR] Failed to read from " TRAIN_LABELS_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    return 0;
}

int mnist_get_testing_sample(size_t index, uint8_t image[IMAGE_SIZE], uint8_t *label)
{
    if (index >= TESTING_SET_LENGTH)
    {
        errno = EINVAL;
        return -1;
    }

    size_t offset_image = T10K_IMAGES_HEADER_SIZE + (index * IMAGE_SIZE);
    size_t offset_label = T10K_LABELS_HEADER_SIZE + index;

    if (fseek(t10k_images_file, offset_image, SEEK_SET) != 0)
    {
        fprintf(stderr, "[ERROR] Failed to navigate " T10K_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fread(image, 1, IMAGE_SIZE, t10k_images_file) != IMAGE_SIZE)
    {
        fprintf(stderr, "[ERROR] Failed to read from " T10K_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fseek(t10k_labels_file, offset_label, SEEK_SET) != 0)
    {
        fprintf(stderr, "[ERROR] Failed to navigate " T10K_IMAGES_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    if (fread(label, 1, 1, t10k_labels_file) != 1)
    {
        fprintf(stderr, "[ERROR] Failed to read from " T10K_LABELS_FILEPATH ": %s\n", strerror(errno));
        return -1;
    }

    return 0;
}

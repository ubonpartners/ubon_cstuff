
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "cuda.h"
#include "image.h"
#include "c_tests.h"
#include "misc.h"
#include "cuda_stuff.h"

int main(int argc, char *argv[])
{
    printf("ubon_cstuff version = %s\n", ubon_cstuff_get_version());
    init_cuda_stuff();
    image_init();

    return run_all_c_tests();
}

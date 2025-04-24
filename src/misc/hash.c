#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include "log.h"

#define HASH_BLOCKSIZE 4096
#define FNV_OFFSET 2166136261u
#define FNV_PRIME 16777619u

void hash_2d(const uint8_t *mem, int w, int h, int stride, uint32_t *dest)
{
    for(int y=0;y<h;y++)
    {
        uint32_t hash = FNV_OFFSET;
        for (int i=0;i<w;i++)
        {
            hash ^= mem[i+y*stride];
            hash *= 16777619u;
        }
        dest[y]=hash;
    }
}

uint32_t hash_u32(const uint32_t *mem, int num)
{
    uint32_t hash = 2166136261u;
    for (int i=0;i<num;i++)
    {
        hash ^= mem[i];
        hash *= 16777619u;
    }
    return hash;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include "log.h"

#define HASH_BLOCKSIZE 4096

static uint32_t hash_block(const uint32_t *mem, int num)
{
    uint32_t hash = 2166136261u;
    for (int i=0;i<num;i++)
    {
        hash ^= mem[i];
        hash *= 16777619u;
    }
    return hash;
}

uint32_t hash_host(void *mem, int size)
{
    int blocks=(size+HASH_BLOCKSIZE-1)/HASH_BLOCKSIZE;
    uint32_t partials[blocks];
    assert((size&3)==0);
    uint8_t *mem8=(uint8_t*)mem;

    for(int i=0;i<blocks;i++)
    {
        partials[i]=hash_block((const uint32_t *)(mem8+HASH_BLOCKSIZE*i), std::min(HASH_BLOCKSIZE, size-HASH_BLOCKSIZE*i)/4);
    }
    return hash_block(partials, blocks);
}

uint32_t hash_2d_host(uint8_t *mem, int w, int h, int stride)
{
    uint32_t partials[h];
    assert((w&3)==0);
    for(int i=0;i<h;i++)
    {
        partials[i]=hash_block((const uint32_t *)(mem+stride*i), w>>2);
    }
    return hash_block(partials, h);
}
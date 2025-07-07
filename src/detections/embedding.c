#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mutex>
#include <float.h>  // for FLT_MIN
#include "assert.h"
#include "memory_stuff.h"
#include "embedding.h"
#include "infer.h"
#include "log.h"
#include "misc.h"

static std::once_flag initFlag;
static block_allocator_t *embedding_allocator;

struct embedding
{
    double time;
    float quality;
    int size;
    void (*wait_fn)(void *context);
    void *wait_fn_context;
    volatile bool ready;
    float data[1];
};

static void embedding_init()
{
    embedding_allocator=block_allocator_create("embedding", sizeof(embedding_t)+128*sizeof(float));
}

embedding_t *embedding_create(int size, double time)
{
    std::call_once(initFlag, embedding_init);
    embedding_t *e=(embedding_t*)block_alloc(embedding_allocator, sizeof(embedding_t)+size*sizeof(float));
    memset(e, 0, sizeof(embedding_t));
    e->ready=true;
    e->size=size;
    return e;
}

void embedding_destroy(embedding_t *e)
{
    //if (block_reference_count(e)==1) embedding_sync(e);
    block_free(e);
}

void embedding_set_wait_function(embedding_t *e, void (*wait)(void *context), void *context)
{
    e->wait_fn=wait;
    e->wait_fn_context=context;
}

void embedding_sync(embedding_t *e)
{
    while(e->ready==false)
    {
        assert(e->wait_fn!=0);
        e->wait_fn(e->wait_fn_context);
    }
}

bool embedding_is_ready(embedding_t *e)
{
    assert(e!=0);
    return e->ready;
}

int embedding_get_size(embedding_t *e)
{
    return e->size;
}

float *embedding_get_data(embedding_t *e)
{
    embedding_sync(e);
    return e->data;
}

void embedding_set_data(embedding_t *e, float *data, int size)
{
    assert(e!=0);
    assert(e->size==size);
    assert(block_reference_count(e)>=1);
    memcpy(e->data, data, sizeof(float)*size);
    e->ready=true;
}

embedding_t *embedding_reference(embedding_t *e)
{
    return (embedding_t *)block_reference(e);
}

void embedding_check(embedding_t *e)
{
    assert(e!=0);
    assert(block_reference_count(e)>=1);
}

void embedding_set_quality(embedding_t *e, float q)
{
    assert(e!=0);
    e->quality=q;
}

float embedding_get_quality(embedding_t *e)
{
    if (e==0) return 0.0f;
    return(e->quality);
}
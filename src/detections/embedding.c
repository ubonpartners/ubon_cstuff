#include <future>
#include <chrono>
#include <mutex>
#include "string.h"
#include "assert.h"
#include "memory_stuff.h"
#include "embedding.h"
#include "infer.h"
#include "log.h"
#include "misc.h"
#include "maths_stuff.h"

static std::once_flag initFlag;
static block_allocator_t *embedding_allocator;

struct embedding
{
    double                   time;
    float                    quality;
    int                      size;

    /*  synchronisation objects are pointers, so no ctors/dtors needed  */
    std::promise<void>*      promise;
    std::shared_future<void>*future;
    bool                     value_set;

    float                    data[1];   /* flexible-array tail */
};

using embedding_t = embedding;

/*------------------------------------------------------------*/
static void embedding_init()
{
    embedding_allocator =
        block_allocator_create("embedding",
                               sizeof(embedding_t) + 128 * sizeof(float));
}

/*------------------------------------------------------------*/
embedding_t *embedding_create(int size, double time)
{
    std::call_once(initFlag, embedding_init);

    /* raw storage */
    embedding_t *e = static_cast<embedding_t *>(
        block_alloc(embedding_allocator,
                    sizeof(embedding_t) + size * sizeof(float)));

    /* zero everything so POD fields start clean */
    memset(e, 0, sizeof(embedding_t));

    /* hand-rolled construction of sync objects */
    e->promise = new std::promise<void>();
    e->future  = new std::shared_future<void>(e->promise->get_future().share());
    e->value_set = false;

    /* metadata */
    e->size = size;
    e->time = time;
    return e;
}

/*------------------------------------------------------------*/
void embedding_destroy(embedding_t *e)
{
    if (block_reference_count(e) == 1)
    {
        e->future->wait();            /* wait for producer */

        /* manual destruction of sync objects */
        delete e->future;
        delete e->promise;
    }
    block_free(e);
}

/*------------------------------------------------------------*/
void embedding_sync(embedding_t *e)
{
    assert(e != nullptr);
    if (e->value_set) return;
    e->future->wait();                /* efficient blocking wait */
    assert(e->value_set);
}

bool embedding_is_ready(embedding_t *e)
{
    assert(e != nullptr);
    return e->value_set;
    //return e->future->wait_for(std::chrono::seconds(0))== std::future_status::ready;
}

/*------------------------------------------------------------*/
int embedding_get_size   (embedding_t *e)
{
    return e->size;
}

float embedding_get_quality(embedding_t *e)
{
    return e ? e->quality : 0.0f;
}

double embedding_get_time (embedding_t *e)
{
    return e->time;
}

float *embedding_get_data(embedding_t *e)
{
    embedding_sync(e);
    return e->data;
}

/*------------------------------------------------------------*/
void embedding_set_data(embedding_t *e, float *src, int size, bool l2_normalize)
{
    assert(e                   != nullptr);
    assert(e->size             == size);
    assert(block_reference_count(e) >= 1);

    assert(e->value_set==false);
    memcpy(e->data, src, sizeof(float) * size);
    if (l2_normalize) vec_l2_norm_inplace(e->data, size);
    /* fulfil promise exactly once */
    e->value_set = true;
    e->promise->set_value();
}

void embedding_set_data_half(embedding_t *e, void *src, int size, bool l2_normalize)
{
    assert(e                   != nullptr);
    assert(e->size             == size);
    assert(block_reference_count(e) >= 1);

    assert(e->value_set==false);

    vec_copy_half_to_float(src, e->data, size);
    if (l2_normalize) vec_l2_norm_inplace(e->data, size);

    /* fulfil promise exactly once */
    e->value_set = true;
    e->promise->set_value();
}

void embedding_set_quality(embedding_t *e, float q)
{
    e->quality = q;
}

void embedding_set_time(embedding_t *e, double t)
{
    e->time= t;
}

embedding_t *embedding_reference(embedding_t *e)
{
    return static_cast<embedding_t *>(block_reference(e));
}

void embedding_check(embedding_t *e)
{
    assert(e != nullptr);
    assert(block_reference_count(e) >= 1);
}

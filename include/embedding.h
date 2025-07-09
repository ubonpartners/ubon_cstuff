#ifndef __EMBEDDING_H
#define __EMBEDDING_H

typedef struct embedding embedding_t;
embedding_t *embedding_create(int size, double time);
void embedding_destroy(embedding_t *e);

void embedding_set_wait_function(embedding_t *e, void (*wait)(void *context), void *context);
void embedding_sync(embedding_t *e);
bool embedding_is_ready(embedding_t *e);
int embedding_get_size(embedding_t *e);
float *embedding_get_data(embedding_t *e);
void embedding_set_data(embedding_t *e, float *d, int size);
embedding_t *embedding_reference(embedding_t *e);
void embedding_check(embedding_t *e);
void embedding_set_quality(embedding_t *e, float q);
float embedding_get_quality(embedding_t *e);
void embedding_set_time(embedding_t *e, double t);
double embedding_get_time(embedding_t *e);

#endif
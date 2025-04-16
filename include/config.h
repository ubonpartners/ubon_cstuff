#ifndef __CONFIG_H
#define __CONFIG_H

typedef struct config config_t;

config_t *config(const char *yaml_file);
void config_destroy(config_t *c);

#endif

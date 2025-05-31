#ifndef __PCAP_DECODER_H
#define __PCAP_DECODER_H

#include "image.h"

typedef struct pcap_decoder pcap_decoder_t;
pcap_decoder_t *pcap_decoder_create(const char *filename);
void pcap_decoder_destroy(pcap_decoder_t *pcap_dec);
image_t *pcap_decoder_get_frame(pcap_decoder_t *pcap_dec);

#endif

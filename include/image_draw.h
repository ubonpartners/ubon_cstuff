#ifndef __IMAGE_DRAW_H
#define __IMAGE_DRAW_H

void image_draw_line(image_t *img, float x0, float y0, float x1, float y1, uint32_t argb);
void image_draw_text(image_t *img, float x_norm, float y_norm, const char *text, uint32_t argb);
void image_draw_box(image_t *img, float x0, float y0, float x1, float y1, uint32_t argb);

#endif

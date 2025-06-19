#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"
#include "image.h"
#include "log.h"

static stbtt_fontinfo g_font;
static int g_font_initialized = 0;
static unsigned char *g_font_buffer = NULL;

static const char *DEFAULT_FONT="/usr/share/fonts/truetype/freefont/FreeMono.ttf";

void image_draw_text(image_t *img, float x_norm, float y_norm, const char *text, uint32_t rgb24)
{
    assert(img->format == IMAGE_FORMAT_RGB24_HOST);
    assert(img->rgb != NULL);

    if (!g_font_initialized) {
        FILE *f = fopen(DEFAULT_FONT, "rb"); // Replace with your TTF font
        if (!f) {
            log_error("Font file %s not found",DEFAULT_FONT);
            return;
        }
        fseek(f, 0, SEEK_END);
        size_t size = ftell(f);
        fseek(f, 0, SEEK_SET);
        g_font_buffer = (unsigned char *)malloc(size);
        assert(size==fread(g_font_buffer, 1, size, f));
        fclose(f);
        if (!stbtt_InitFont(&g_font, g_font_buffer, 0)) {
            fprintf(stderr, "Failed to init font.\n");
            return;
        }
        g_font_initialized = 1;
    }

    float font_height = 16.0f;
    float scale = stbtt_ScaleForPixelHeight(&g_font, font_height);

    int x_pix = (int)(x_norm * img->width);
    int y_pix = (int)(y_norm * img->height);

    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&g_font, &ascent, &descent, &lineGap);
    int baseline = y_pix + (int)(ascent * scale);

    for (const char *p = text; *p; p++) {
        int glyph = stbtt_FindGlyphIndex(&g_font, *p);
        int w, h, xoff, yoff;
        unsigned char *bitmap = stbtt_GetGlyphBitmap(&g_font, scale, scale, glyph, &w, &h, &xoff, &yoff);

        for (int j = 0; j < h; j++) {
            for (int i = 0; i < w; i++) {
                int xi = x_pix + xoff + i;
                int yi = baseline + yoff + j;
                if (xi < 0 || yi < 0 || xi >= img->width || yi >= img->height) continue;
                uint8_t alpha = bitmap[j * w + i];
                uint8_t *px = &img->rgb[xi * 3 + yi * img->stride_rgb];

                // Alpha blend (src over dst)
                for (int k = 0; k < 3; k++) {
                    uint8_t src = (rgb24 >> (16 - 8 * k)) & 0xFF;
                    px[k] = (uint8_t)((alpha * src + (255 - alpha) * px[k]) / 255);
                }
            }
        }

        int advance;
        stbtt_GetGlyphHMetrics(&g_font, glyph, &advance, 0);
        x_pix += (int)(advance * scale);

        stbtt_FreeBitmap(bitmap, NULL);
    }
}


void image_draw_line(image_t *img, float x0, float y0, float x1, float y1, int clr)
{
    int steps=(int)(fmaxf(fabsf(x1-x0)*img->width, fabsf(y1-y0)*img->height)+0.999);
    if (steps==0) return;
    assert(img->format==IMAGE_FORMAT_RGB24_HOST);
    assert(img->rgb!=0);
    for(int i=0;i<=steps;i++)
    {
        float l=(i+0.0)/(steps+0.0);
        float x=x0*(1.0-l)+x1*l;
        float y=y0*(1.0-l)+y1*l;
        int xi=(int)(x*img->width);
        int yi=(int)(y*img->height);
        if ((xi<0)||(yi<0)||(xi>=img->width)||(yi>=img->height)) continue;
        img->rgb[xi*3+yi*img->stride_rgb+0]=(clr>>16)&0xff;
        img->rgb[xi*3+yi*img->stride_rgb+1]=(clr>>8)&0xff;
        img->rgb[xi*3+yi*img->stride_rgb+2]=(clr>>0)&0xff;
    }
}
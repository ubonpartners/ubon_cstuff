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

void image_draw_text(image_t *img, float x_norm, float y_norm, const char *text, uint32_t argb)
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

    float font_height = 18.0f;
    float scale = stbtt_ScaleForPixelHeight(&g_font, font_height);

    int x_pix = (int)(x_norm * img->width);
    int y_pix = (int)(y_norm * img->height);

    int ascent, descent, lineGap;
    stbtt_GetFontVMetrics(&g_font, &ascent, &descent, &lineGap);
    int baseline = y_pix + (int)(ascent * scale);

    int bb_x_min=10000;
    int bb_x_max=0;
    int bb_y_min=10000;
    int bb_y_max=0;
    int b_x_pix=x_pix;
    for (const char *p = text; *p; p++) {
        int glyph = stbtt_FindGlyphIndex(&g_font, *p);
        int w, h, xoff, yoff;
        unsigned char *bitmap = stbtt_GetGlyphBitmap(&g_font, scale, scale, glyph, &w, &h, &xoff, &yoff);

        int xmin=b_x_pix+xoff+0;
        int xmax=b_x_pix+xoff+w;
        int ymin=baseline+yoff+0;
        int ymax=baseline+yoff+h;
        bb_x_min=std::min(bb_x_min, xmin);
        bb_y_min=std::min(bb_y_min, ymin);
        bb_x_max=std::max(bb_x_max, xmax);
        bb_y_max=std::max(bb_y_max, ymax);

        int advance;
        stbtt_GetGlyphHMetrics(&g_font, glyph, &advance, 0);
        b_x_pix += (int)(advance * scale);

        stbtt_FreeBitmap(bitmap, NULL);
    }

    for (int j = bb_y_min; j < bb_y_max; j++) {
        for (int i = bb_x_min; i < bb_x_max; i++) {
            uint8_t *px = &img->rgb[i * 3 + j * img->stride_rgb];
            int a=190;
            px[0]=(px[0]*a)/256;
            px[1]=(px[1]*a)/256;
            px[2]=(px[2]*a)/256;
        }
    }

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
                    uint8_t src = (argb >> (16 - 8 * k)) & 0xFF;
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


void image_draw_line(image_t *img, float x0, float y0, float x1, float y1, uint32_t argb)
{
    int steps=(int)(fmaxf(fabsf(x1-x0)*img->width, fabsf(y1-y0)*img->height)+0.999);
    if (steps==0) return;
    assert(img->format==IMAGE_FORMAT_RGB24_HOST);
    assert(img->rgb!=0);
    uint32_t a=argb>>24;
    for(int i=0;i<=steps;i++)
    {
        float l=(i+0.0)/(steps+0.0);
        float x=x0*(1.0-l)+x1*l;
        float y=y0*(1.0-l)+y1*l;
        int xi=(int)(x*img->width);
        int yi=(int)(y*img->height);
        if ((xi<0)||(yi<0)||(xi>=img->width)||(yi>=img->height)) continue;
        int offs=xi*3+yi*img->stride_rgb;
        img->rgb[offs+0]=((img->rgb[offs+0]*(256-a)+(argb>>16)*a))>>8;
        img->rgb[offs+1]=((img->rgb[offs+1]*(256-a)+(argb>> 8)*a))>>8;
        img->rgb[offs+2]=((img->rgb[offs+2]*(256-a)+(argb>> 0)*a))>>8;
    }
}

void image_draw_box(image_t *img, float x0, float y0, float x1, float y1, uint32_t argb)
{
    if (img==0) return;
    assert(img->format==IMAGE_FORMAT_RGB24_HOST);
    image_draw_line(img, x0, y0, x1, y0, argb);
    image_draw_line(img, x0, y1, x1, y1, argb);
    image_draw_line(img, x0, y0, x0, y1, argb);
    image_draw_line(img, x1, y0, x1, y1, argb);
}

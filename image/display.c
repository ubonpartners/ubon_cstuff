#include <SDL2/SDL.h>
#include <stdio.h>
#include "display.h"

struct display
{
    char title[128];
    int width, height;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
};

display_t *display_create(const char *title)
{
    display_t *d=(display_t *)malloc(sizeof(display_t));
    memset(d, 0, sizeof(display_t));
    strncpy(d->title, title, 100);
    return d;
}

void display_destroy(display_t *d)
{
    if (!d) return;
    if (d->texture) SDL_DestroyTexture(d->texture);
    if (d->renderer) SDL_DestroyRenderer(d->renderer);
    if (d->window) SDL_DestroyWindow(d->window);
    free(d);
}


void display_image(const char *txt, image_t *img)
{
    static display_t *hack_display=0;
    if (hack_display==0) hack_display=display_create(txt);
    display_image(hack_display, img);
}

void display_image(display_t *d, image_t *img)
{
    if (img->format!=IMAGE_FORMAT_RGB24_HOST)
    {
        image_t *tmp=image_convert(img, IMAGE_FORMAT_RGB24_HOST);
        display_image(d, tmp);
        destroy_image(tmp);
        return;
    }
    int s=2;
    if (d->window!=0)
    {
        if ((d->width!=img->width*s)||(d->height!=img->height*s))
        {
            if (d->texture) SDL_DestroyTexture(d->texture);
            if (d->renderer) SDL_DestroyRenderer(d->renderer);
            if (d->window) SDL_DestroyWindow(d->window);
            d->texture=0;
            d->renderer=0;
            d->window=0;
        }
    }
    if (d->window==0)
    {
        d->width=img->width*s;
        d->height=img->height*s;
        d->window = SDL_CreateWindow(d->title, SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,img->width*s, img->height*s,SDL_WINDOW_SHOWN);  
        d->renderer = SDL_CreateRenderer(d->window, -1, SDL_RENDERER_ACCELERATED);
        d->texture = SDL_CreateTexture(d->renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STREAMING,img->width,img->height);
    }

    bool record=false;
    if (record)
    {
        static FILE *recf=0;
        if (recf==0)
        {
            char temp[256];
            sprintf(temp,"rec%dx%d.rgb",img->width, img->height);
            recf=fopen(temp,"wb");
        }
        if (recf)
        {
            fwrite(img->rgb, 1, img->width*img->height*3, recf);
            fflush(recf);
        }
    }

    image_sync(img);
    SDL_UpdateTexture(d->texture, NULL, img->rgb, img->stride_rgb);
    SDL_RenderClear(d->renderer);
    SDL_RenderCopy(d->renderer, d->texture, NULL, NULL);
    SDL_RenderPresent(d->renderer);

    SDL_Event e;
    SDL_PollEvent(&e);
    if (e.type == SDL_QUIT) 
    {
        printf("Quitting\n");
        exit(-1);
    }
}

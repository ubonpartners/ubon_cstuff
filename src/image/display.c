#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <time.h>
#include "display.h"
#include "log.h"

/*
 * Threaded SDL display with:
 *  • Resizable windows (SDL_WINDOW_RESIZABLE)
 *  • Automatic re-rendering on resize (using the last image reference)
 *  • One SDL thread that wakes every 10 ms (or on new command)
 *  • Polls exactly one SDL event per iteration to keep responsiveness
 *
 * We store, inside each display_t:
 *   • img_width/img_height: the last image’s native dimensions
 *   • current_img:        the last image_t* (with reference held)
 *   • window, renderer, texture
 *
 * On each display command, we replace current_img (destroying old reference),
 * create or update the texture, upload pixels, and render into the window.
 *
 * In the SDL event loop, if we see SDL_WINDOWEVENT_RESIZED, we:
 *   • Look up the display_t* via window data
 *   • Recompute the destination rectangle for letterboxing/pillarboxing
 *   • Re-render using current_img
 *
 * Public API (display.h):
 *   typedef struct display display_t;
 *   display_t *display_create(const char *title);
 *   void        display_destroy(display_t *d);
 *   void        display_image(const char *name, image_t *img);
 *   void        display_image(display_t *d, image_t *img);
 */

struct display {
    char          title[128];
    int           img_width;    /* Last image’s native width */
    int           img_height;   /* Last image’s native height */
    image_t      *current_img;  /* Reference to last image (NULL if none) */
    SDL_Window   *window;
    SDL_Renderer *renderer;
    SDL_Texture  *texture;      /* Always matches img_width × img_height */
};

/* Command types sent to the SDL thread */
typedef enum {
    CMD_DISPLAY_NAMED_IMAGE,  /* name + image_t* */
    CMD_DISPLAY_IMAGE,        /* existing display_t* + image_t* */
    CMD_DESTROY_DISPLAY       /* existing display_t* */
} cmd_type_t;

/* A node in the command queue */
typedef struct cmd_node {
    cmd_type_t       type;
    char            *name;   /* Used only for CMD_DISPLAY_NAMED_IMAGE */
    image_t         *img;    /* Used by both display commands */
    display_t       *d;      /* Used by CMD_DISPLAY_IMAGE & CMD_DESTROY_DISPLAY */
    struct cmd_node *next;
} cmd_node_t;

/* Global queue head/tail, with mutex and condition variable */
static cmd_node_t      *g_queue_head = NULL;
static cmd_node_t      *g_queue_tail = NULL;
static pthread_mutex_t  g_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_queue_cond  = PTHREAD_COND_INITIALIZER;

/* SDL thread handle & startup flag */
static pthread_t  g_sdl_thread;
static int        g_sdl_thread_started = 0;

/* Forward declarations */
static void *sdl_thread_loop(void *arg);
static void enqueue_command(cmd_node_t *cmd);

/* Start the SDL thread on first use */
static void init_sdl_thread_if_needed(void) {
    if (!g_sdl_thread_started) {
        if (pthread_create(&g_sdl_thread, NULL, sdl_thread_loop, NULL) != 0) {
            log_error("Failed to create SDL thread");
            exit(1);
        }
        pthread_detach(g_sdl_thread);
        g_sdl_thread_started = 1;
    }
}

/* ================= Public API ================= */

/* Create a bare display_t (for manual use). */
display_t *display_create(const char *title) {
    init_sdl_thread_if_needed();

    display_t *d = (display_t *)malloc(sizeof(display_t));
    memset(d, 0, sizeof(display_t));
    strncpy(d->title, title, sizeof(d->title) - 1);
    d->title[sizeof(d->title) - 1] = '\0';
    d->img_width    = 0;
    d->img_height   = 0;
    d->current_img  = NULL;
    d->window       = NULL;
    d->renderer     = NULL;
    d->texture      = NULL;
    return d;
}

/* Destroy a display_t* (manual). */
void display_destroy(display_t *d) {
    if (!d) return;

    cmd_node_t *cmd = (cmd_node_t *)malloc(sizeof(cmd_node_t));
    cmd->type = CMD_DESTROY_DISPLAY;
    cmd->d    = d;
    cmd->img  = NULL;
    cmd->name = NULL;
    cmd->next = NULL;
    enqueue_command(cmd);
}

/* “Fire & forget” by name: like cv2::imshow(window_name, img). */
void display_image(const char *name, image_t *img) {
    init_sdl_thread_if_needed();

    image_t *ref = image_reference(img);
    char    *name_copy = strdup(name);

    cmd_node_t *cmd = (cmd_node_t *)malloc(sizeof(cmd_node_t));
    cmd->type = CMD_DISPLAY_NAMED_IMAGE;
    cmd->name = name_copy;
    cmd->img  = ref;
    cmd->d    = NULL;
    cmd->next = NULL;
    enqueue_command(cmd);
}

/* Explicit display_t* + image_t*. */
void display_image(display_t *d, image_t *img) {
    if (!d || !img) return;
    init_sdl_thread_if_needed();

    image_t *ref = image_reference(img);

    cmd_node_t *cmd = (cmd_node_t *)malloc(sizeof(cmd_node_t));
    cmd->type = CMD_DISPLAY_IMAGE;
    cmd->d    = d;
    cmd->img  = ref;
    cmd->name = NULL;
    cmd->next = NULL;
    enqueue_command(cmd);
}

/* Enqueue a command and signal the SDL thread */
static void enqueue_command(cmd_node_t *cmd) {
    pthread_mutex_lock(&g_queue_mutex);
    if (g_queue_tail) {
        g_queue_tail->next = cmd;
        g_queue_tail = cmd;
    } else {
        g_queue_head = g_queue_tail = cmd;
    }
    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_mutex);
}

/* ================= SDL Thread Data Structures ================= */

/* Entry for a named display (managed only on the SDL thread) */
typedef struct named_display_entry {
    char                       *name;  /* strdup’d */
    display_t                  *d;
    struct named_display_entry *next;
} named_display_entry_t;

/* Head of the linked list of named displays */
static named_display_entry_t *g_named_list = NULL;

/* Look up or create a display_t* for a given name, using the incoming image’s dimensions */
static display_t *lookup_or_create_named_display(const char *name, image_t *img) {
    for (named_display_entry_t *e = g_named_list; e; e = e->next) {
        if (strcmp(e->name, name) == 0) {
            return e->d;
        }
    }
    /* Not found → create a new display_t with a title that includes size/format */
    char titlebuf[128];
    snprintf(titlebuf, sizeof(titlebuf) - 1,
             "%s : %dx%d : %s",
             name, img->width, img->height,
             image_format_name(img->format));

    display_t *new_display = (display_t *)malloc(sizeof(display_t));
    memset(new_display, 0, sizeof(display_t));
    strncpy(new_display->title, titlebuf, sizeof(new_display->title) - 1);
    new_display->title[sizeof(new_display->title) - 1] = '\0';
    new_display->img_width    = 0;
    new_display->img_height   = 0;
    new_display->current_img  = NULL;
    new_display->window       = NULL;
    new_display->renderer     = NULL;
    new_display->texture      = NULL;

    named_display_entry_t *entry = (named_display_entry_t *)malloc(sizeof(*entry));
    entry->name = strdup(name);
    entry->d    = new_display;
    entry->next = g_named_list;
    g_named_list = entry;

    log_debug("SDL thread: created named display \"%s\"", name);
    return new_display;
}

/* Clean up one named‐display entry: free name, destroy SDL objects, free display_t, free entry */
static void destroy_named_entry(named_display_entry_t *prev, named_display_entry_t *e) {
    if (prev) {
        prev->next = e->next;
    } else {
        g_named_list = e->next;
    }
    free(e->name);

    if (e->d->texture) SDL_DestroyTexture(e->d->texture);
    if (e->d->renderer) SDL_DestroyRenderer(e->d->renderer);
    if (e->d->window) SDL_DestroyWindow(e->d->window);
    if (e->d->current_img) destroy_image(e->d->current_img);
    free(e->d);

    free(e);
}

/* Helper: (re)render the display_t using its current_img, preserving aspect ratio */
static void render_display_with_current_image(display_t *d) {
    if (!d || !d->current_img) return;

    /* Compute letterboxed destination rectangle based on current window size */
    int win_w, win_h;
    SDL_GetWindowSize(d->window, &win_w, &win_h);

    float img_aspect = (float)d->img_width / (float)d->img_height;
    float win_aspect = (float)win_w / (float)win_h;

    SDL_Rect dst_rect;
    if (win_aspect > img_aspect) {
        /* Window is wider than image: letterbox horizontally */
        dst_rect.h = win_h;
        dst_rect.w = (int)(win_h * img_aspect);
        dst_rect.x = (win_w - dst_rect.w) / 2;
        dst_rect.y = 0;
    } else {
        /* Window is taller (or equal): letterbox vertically */
        dst_rect.w = win_w;
        dst_rect.h = (int)(win_w / img_aspect);
        dst_rect.x = 0;
        dst_rect.y = (win_h - dst_rect.h) / 2;
    }

    SDL_RenderClear(d->renderer);
    SDL_RenderCopy(d->renderer, d->texture, NULL, &dst_rect);
    SDL_RenderPresent(d->renderer);
}

/* ================= SDL Thread Main Loop ================= */

static void *sdl_thread_loop(void *arg) {
    (void)arg;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        log_error("SDL_Init failed: %s", SDL_GetError());
        exit(1);
    }

    while (1) {
        /* Wait with a short timeout (10 ms) so we can poll SDL events regularly */
        struct timespec now, timeout_ts;
        clock_gettime(CLOCK_REALTIME, &now);
        timeout_ts.tv_sec  = now.tv_sec;
        timeout_ts.tv_nsec = now.tv_nsec + 10 * 1000000;  // +10 ms
        if (timeout_ts.tv_nsec >= 1000000000) {
            timeout_ts.tv_sec += 1;
            timeout_ts.tv_nsec -= 1000000000;
        }

        pthread_mutex_lock(&g_queue_mutex);
        if (!g_queue_head) {
            pthread_cond_timedwait(&g_queue_cond, &g_queue_mutex, &timeout_ts);
        }
        /* Drain all pending commands */
        while (g_queue_head) {
            cmd_node_t *cmd = g_queue_head;
            g_queue_head = cmd->next;
            if (!g_queue_head) {
                g_queue_tail = NULL;
            }
            pthread_mutex_unlock(&g_queue_mutex);

            if (cmd->type == CMD_DISPLAY_NAMED_IMAGE) {
                /* Lookup/create display_t by name */
                display_t *d       = lookup_or_create_named_display(cmd->name, cmd->img);
                char     *name_for_free = cmd->name;
                image_t  *img_for_use   = cmd->img;

                /* Convert to RGB24 if needed */
                if (img_for_use->format != IMAGE_FORMAT_RGB24_HOST) {
                    image_t *tmp = image_convert(img_for_use, IMAGE_FORMAT_RGB24_HOST);
                    destroy_image(img_for_use);
                    img_for_use = tmp;
                }

                /* Replace current_img: destroy old reference if any */
                if (d->current_img) {
                    destroy_image(d->current_img);
                }
                d->current_img = img_for_use;
                d->img_width   = img_for_use->width;
                d->img_height  = img_for_use->height;

                /* If window doesn’t exist yet, create it as resizable */
                if (!d->window) {
                    /* Initial window size = img_width×2, img_height×2 */
                    int win_w = std::max(1920, d->img_width  * 2);
                    int win_h = std::max(1080, d->img_height * 2);

                    d->window = SDL_CreateWindow(
                        d->title,
                        SDL_WINDOWPOS_UNDEFINED,
                        SDL_WINDOWPOS_UNDEFINED,
                        win_w,
                        win_h,
                        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
                    );
                    d->renderer = SDL_CreateRenderer(d->window, -1, SDL_RENDERER_ACCELERATED);

                    /* Associate display_t* with SDL_Window via window data */
                    SDL_SetWindowData(d->window, "display", d);

                    d->texture = SDL_CreateTexture(
                        d->renderer,
                        SDL_PIXELFORMAT_RGB24,
                        SDL_TEXTUREACCESS_STREAMING,
                        d->img_width,
                        d->img_height
                    );
                }
                else {
                    /* Window exists: ensure texture matches new image size */
                    int tex_w, tex_h;
                    SDL_QueryTexture(d->texture, NULL, NULL, &tex_w, &tex_h);
                    if (tex_w != d->img_width || tex_h != d->img_height) {
                        SDL_DestroyTexture(d->texture);
                        d->texture = SDL_CreateTexture(
                            d->renderer,
                            SDL_PIXELFORMAT_RGB24,
                            SDL_TEXTUREACCESS_STREAMING,
                            d->img_width,
                            d->img_height
                        );
                    }
                }

                /* Upload pixels into the (native‐size) texture */
                image_sync(d->current_img);
                SDL_UpdateTexture(d->texture, NULL,
                                  d->current_img->rgb,
                                  d->current_img->stride_rgb);

                /* Render once initially */
                render_display_with_current_image(d);

                free(name_for_free);
            }
            else if (cmd->type == CMD_DISPLAY_IMAGE) {
                display_t *d         = cmd->d;
                image_t   *img_for_use = cmd->img;

                /* Convert to RGB24 if needed */
                if (img_for_use->format != IMAGE_FORMAT_RGB24_HOST) {
                    image_t *tmp = image_convert(img_for_use, IMAGE_FORMAT_RGB24_HOST);
                    destroy_image(img_for_use);
                    img_for_use = tmp;
                }

                /* Replace current_img: destroy old reference if any */
                if (d->current_img) {
                    destroy_image(d->current_img);
                }
                d->current_img = img_for_use;
                d->img_width   = img_for_use->width;
                d->img_height  = img_for_use->height;

                /* If window doesn’t exist yet, create it as resizable */
                if (!d->window) {
                    int win_w = d->img_width  * 2;
                    int win_h = d->img_height * 2;

                    d->window = SDL_CreateWindow(
                        d->title,
                        SDL_WINDOWPOS_UNDEFINED,
                        SDL_WINDOWPOS_UNDEFINED,
                        win_w,
                        win_h,
                        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
                    );
                    d->renderer = SDL_CreateRenderer(d->window, -1, SDL_RENDERER_ACCELERATED);
                    SDL_SetWindowData(d->window, "display", d);

                    d->texture = SDL_CreateTexture(
                        d->renderer,
                        SDL_PIXELFORMAT_RGB24,
                        SDL_TEXTUREACCESS_STREAMING,
                        d->img_width,
                        d->img_height
                    );
                }
                else {
                    /* Window exists: ensure texture matches new image size */
                    int tex_w, tex_h;
                    SDL_QueryTexture(d->texture, NULL, NULL, &tex_w, &tex_h);
                    if (tex_w != d->img_width || tex_h != d->img_height) {
                        SDL_DestroyTexture(d->texture);
                        d->texture = SDL_CreateTexture(
                            d->renderer,
                            SDL_PIXELFORMAT_RGB24,
                            SDL_TEXTUREACCESS_STREAMING,
                            d->img_width,
                            d->img_height
                        );
                    }
                }

                /* Upload pixels */
                image_sync(d->current_img);
                SDL_UpdateTexture(d->texture, NULL,
                                  d->current_img->rgb,
                                  d->current_img->stride_rgb);

                /* Render once */
                render_display_with_current_image(d);
            }
            else if (cmd->type == CMD_DESTROY_DISPLAY) {
                display_t *d = cmd->d;

                /* Remove from named list if present */
                named_display_entry_t *prev = NULL;
                for (named_display_entry_t *e = g_named_list; e; e = e->next) {
                    if (e->d == d) {
                        destroy_named_entry(prev, e);
                        break;
                    }
                    prev = e;
                }
                /* If not found in named list, treat as manual display */
                if (!prev || (prev == NULL && (g_named_list == NULL || g_named_list->d != d))) {
                    if (d->texture) SDL_DestroyTexture(d->texture);
                    if (d->renderer) SDL_DestroyRenderer(d->renderer);
                    if (d->window) SDL_DestroyWindow(d->window);
                    if (d->current_img) destroy_image(d->current_img);
                    free(d);
                }
            }

            free(cmd);
            pthread_mutex_lock(&g_queue_mutex);
        }
        pthread_mutex_unlock(&g_queue_mutex);

        /* Poll exactly one SDL event per iteration to keep windows responsive */
        SDL_Event e;
        if (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                printf("Quitting\n");
                exit(-1);
            }
            else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) {
                /* User resized a window: re-render that display’s last image */
                SDL_Window  *w = SDL_GetWindowFromID(e.window.windowID);
                if (w) {
                    display_t *d = (display_t *)SDL_GetWindowData(w, "display");
                    if (d && d->current_img) {
                        /* Recreate the destination rectangle and render */
                        render_display_with_current_image(d);
                    }
                }
            }
            /* (Other events can be handled here if desired) */
        }
        /* Loop back, waiting up to 10 ms again */
    }

    /* Unreachable, but for completeness: */
    SDL_Quit();
    return NULL;
}

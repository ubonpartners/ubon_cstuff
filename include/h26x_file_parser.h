// h26x_file_parser.h

#ifndef H26X_FILE_PARSER_H
#define H26X_FILE_PARSER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "h26x_assembler.h"
// We re‐use the following definitions from h26x_assembler.h:
//   • h26x_codec_t
//   • h26x_nal_stats_t
//   • h26x_frame_descriptor_t
//   • h26x_frame_callback_fn

// Opaque parser handle
typedef struct h26x_file_parser h26x_file_parser_t;

/**
 * Create a new H.264/H.265 file parser.
 *
 * @param codec   Either H26X_CODEC_H264 or H26X_CODEC_H265.
 * @param context User‐provided pointer; passed back verbatim to each callback.
 * @param cb      Callback function to invoke once per assembled frame.
 *
 * @return A pointer to a newly allocated parser, or NULL on failure.
 */
h26x_file_parser_t *
h26x_file_parser_create(h26x_codec_t            codec,
                        void                   *context,
                        h26x_frame_callback_fn  cb);

/**
 * Destroy a file parser. Frees all internal buffers. After calling this,
 * the parser handle is invalid and must not be used.
 */
void
h26x_file_parser_destroy(h26x_file_parser_t *parser);

/**
 * Reset the parser’s internal state (discard any partially‐assembled frame).
 * You can call this if you want to parse another buffer in the same instance.
 */
void
h26x_file_parser_reset(h26x_file_parser_t *parser);

/**
 * Parse an entire Annex‐B buffer (H.264/H.265) and invoke the callback once
 * per assembled frame (“access unit”).  The supplied @framerate should be the
 * nominal frames‐per‐second in the file.  We compute each frame’s 90 kHz
 * timestamp as: timestamp = frame_index * (90000 / framerate), starting at 0.
 *
 * We now detect both 4-byte (0x00000001) and 3-byte (0x000001) start codes.
 *
 * @param parser    The parser returned by h26x_file_parser_create().
 * @param data      Pointer to the full Annex‐B buffer (must remain valid).
 * @param length    Number of bytes in @data.
 * @param framerate Nominal frames per second (e.g. 29.97f or 25.0f).
 */
void
h26x_file_parser_parse_buffer(h26x_file_parser_t *parser,
                              const uint8_t      *data,
                              size_t              length,
                              float               framerate);


#endif // H26X_FILE_PARSER_H

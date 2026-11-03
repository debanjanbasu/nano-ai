#include <stdbool.h>
#include <stdint.h>

/// JSON report for one prompt (see micro-ios/src/lib.rs); free it with `micro_ai_free`.
char *micro_ai_run(const char *model_dir, const char *prompt, const char *image_path,
                   uint32_t max_tokens);
void micro_ai_free(char *report);
/// Prefill the prompt up to the user's text so far ("" at launch: open the engine, pin the
/// system turn), so the next `micro_ai_run` only prefills the rest.
bool micro_ai_warm(const char *model_dir, const char *typed);

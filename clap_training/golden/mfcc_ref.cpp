// Reference features from the board's own front end, built natively:
// the KWS sketch's DC blocker and quantizer (copied, see the .ino) around its mfcc.cpp.
// stdin: raw int16 LE PCM. stdout: uint8 features, 10 per frame, 640-sample frames at a 320 hop.
#include <cstdint>
#include <cstdio>
#include <vector>
#include "mfcc.h"

static int32_t prev_in = 0, prev_out = 0;

static int16_t dc_block(int16_t x) {  // remove_dc_offset() in the .ino
  int32_t out = x - prev_in + ((32700 * prev_out) >> 15);
  prev_in = x;
  prev_out = out;
  if (out > 32767) out = 32767;
  if (out < -32768) out = -32768;
  return static_cast<int16_t>(out);
}

static uint8_t quantize(float c) {  // quantize_feature() in the .ino
  float s = ((c / 123.56967163085938f) + 1.0f) * 128.0f;
  if (s < 0.0f) s = 0.0f;
  else if (s > 255.0f) s = 255.0f;
  return static_cast<uint8_t>(s);
}

int main() {
  if (!mfcc_begin()) return 1;
  std::vector<int16_t> pcm;
  int16_t s;
  while (fread(&s, 2, 1, stdin) == 1) pcm.push_back(dc_block(s));
  float c[kMfccFeatures];
  for (size_t start = 0; start + kMfccFrameSamples <= pcm.size(); start += 320) {
    mfcc_compute(&pcm[start], c);
    for (int i = 0; i < kMfccFeatures; ++i) putchar(quantize(c[i]));
  }
  return 0;
}

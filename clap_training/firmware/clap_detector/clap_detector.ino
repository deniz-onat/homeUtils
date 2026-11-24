// Double-clap detector for Nicla Vision + BrainBoard1500 (AKD1500).
//
// Adapted from the BB15 library's bb15_nicla_vision_keyword_spotting example
// (MIT, see LICENSE-BB15): same microphone chain, DC blocker and MFCC; the model
// is clap_training's fine-tuned DS-CNN (program.cpp, from export_bb15.py).
// Differences from the example, all to match how clap_training evaluated it:
//   - ungated: features and inference run on every 60 ms block, no RMS gate,
//     and the spectrogram is never cleared (training saw continuous audio);
//   - features quantized with the remap in feature_constants.h, not the
//     example's quantize_feature();
//   - decision: smoothed score (alpha 0.7), kStreak scores in a row at or over
//     kThreshold, then kRefractoryMs without inference.
//
// Serial (USB CDC), one text line each:
//   board -> host  "DOUBLECLAP score=0.973 ms=123456" on every detection,
//                  "[clap] ..." boot, status and heartbeat lines,
//                  "S <score> <smoothed>" per inference while SCORES is on,
//                  "P <hex>" / "F <hex>" per block during DUMP, then "DUMP END".
//   host -> board  "SCORES" toggles score lines; "DUMP" restarts the front end
//                  and streams kDumpBlocks blocks of raw PCM and features for
//                  board_check.py.

#include <Arduino.h>
#include <BB15.h>
#include <PDM.h>
#include <math.h>

#include "akida/program_info.h"
#include "feature_constants.h"
#include "mfcc.h"
#include "model_metadata.h"
#include "program.h"

#ifndef ARDUINO_NICLA_VISION
#error "clap_detector requires Arduino Nicla Vision."
#endif

namespace {

constexpr uint32_t kSerialWaitMs = 3000u;
constexpr uint32_t kHeartbeatMs = 30000u;
constexpr size_t kPdmBufferBytes = 1024u;  // as nicla_utils/recorder
constexpr size_t kPdmBufferSamples = kPdmBufferBytes / sizeof(int16_t);
constexpr int kPdmGain = 12;  // every training recording used 12
constexpr uint32_t kSampleRateHz = 16000u;
constexpr uint16_t kHopSamples = 320u;
constexpr uint16_t kBlockSamples = 960u;  // 60 ms, one inference
constexpr uint8_t kFramesPerBlock = kBlockSamples / kHopSamples;
constexpr uint16_t kFrames = 49u;
constexpr uint8_t kCoefficients = 10u;
constexpr uint8_t kClassCount = 2u;  // 0 negative, 1 double_clap
constexpr uint8_t kClapClass = 1u;
constexpr int32_t kDcBlockAlphaQ15 = 32700;  // the example's DC blocker
constexpr uint32_t kAkidaSpiClockHz = 25000000u;

// Decision logic, as clap_training/evaluate.py (threshold chosen on val).
constexpr float kAlpha = 0.7f;
constexpr float kThreshold = 0.95f;
constexpr uint8_t kStreak = 3u;
constexpr uint32_t kRefractoryMs = 1500u;

constexpr uint16_t kDumpBlocks = 50u;  // 3 s

enum class Led : uint8_t { Off, Red, Green, Blue };

struct DcBlockState {
  int32_t previousInput = 0;
  int32_t previousOutput = 0;
};

int16_t g_pdm_samples[kPdmBufferSamples];
volatile bool g_pdm_ready = false;
volatile uint32_t g_pdm_overruns = 0u;

// [0, hop) is the previous block's last hop, [hop, hop + block) this block.
int16_t g_mfcc_input[kHopSamples + kBlockSamples];
int16_t g_raw_block[kBlockSamples];  // before the DC blocker, for DUMP
size_t g_block_fill = 0u;
DcBlockState g_dc_block;

// Ring of quantized frames; g_ring_index is the oldest frame.
uint8_t g_ring[kFrames * kCoefficients];
uint16_t g_ring_index = 0u;
uint16_t g_frames_seen = 0u;  // inference starts once the ring is full
uint8_t g_block_features[kFramesPerBlock * kCoefficients];
uint8_t g_model_input[kFrames * kCoefficients];

float g_smoothed = 0.0f;
uint8_t g_streak = 0u;
uint32_t g_quiet_until_ms = 0u;
uint32_t g_inferences = 0u;
uint32_t g_detections = 0u;
bool g_print_scores = false;
uint16_t g_dump_left = 0u;
bool g_ready = false;
uint32_t g_next_heartbeat_ms = 0u;

BB15Pinout g_pinout = BB15Pinout::niclaVisionDefaults();
BB15Config g_config = []() {
  BB15Config config = BB15Config::defaults();
  config.spiClockHz = kAkidaSpiClockHz;
  return config;
}();
BB15* g_bb15 = nullptr;
BB15Runner* g_runner = nullptr;
BB15Model g_model(program, static_cast<size_t>(program_len));
const int32_t* g_output_shifts = nullptr;
const float* g_output_scales = nullptr;

void set_led(Led color) {  // active low
  digitalWrite(LEDR, color == Led::Red ? LOW : HIGH);
  digitalWrite(LEDG, color == Led::Green ? LOW : HIGH);
  digitalWrite(LEDB, color == Led::Blue ? LOW : HIGH);
}

void remove_dc_offset(int16_t* samples, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    int32_t out = samples[i] - g_dc_block.previousInput +
                  ((kDcBlockAlphaQ15 * g_dc_block.previousOutput) >> 15);
    g_dc_block.previousInput = samples[i];
    g_dc_block.previousOutput = out;
    if (out > 32767) out = 32767;
    if (out < -32768) out = -32768;
    samples[i] = static_cast<int16_t>(out);
  }
}

// frontend.quantize_akida(): float32 throughout, separate multiply and add
// (no FMA, as numpy), round half to even (nearbyintf in the default mode).
__attribute__((optimize("fp-contract=off"))) uint8_t quantize_akida(float c, uint8_t index) {
  const float tf_like = c * kAkidaScale[index] + kAkidaOffset[index];
  const float scaled = (tf_like - kAkidaMfccMin) / kAkidaMfccRange * 255.0f;
  const float rounded = nearbyintf(scaled);
  if (rounded < 0.0f) return 0u;
  if (rounded > 255.0f) return 255u;
  return static_cast<uint8_t>(rounded);
}

void extract_features() {
  float coefficients[kMfccFeatures];
  for (uint8_t frame = 0u; frame < kFramesPerBlock; ++frame) {
    mfcc_compute(&g_mfcc_input[frame * kHopSamples], coefficients);
    uint8_t* slot = &g_ring[g_ring_index * kCoefficients];
    for (uint8_t i = 0u; i < kCoefficients; ++i) {
      slot[i] = quantize_akida(coefficients[i], i);
      g_block_features[frame * kCoefficients + i] = slot[i];
    }
    g_ring_index = (g_ring_index + 1u) % kFrames;
    if (g_frames_seen < kFrames) ++g_frames_seen;
  }
}

// Returns the double_clap probability, or a negative value on failure.
float infer() {
  for (uint16_t frame = 0u; frame < kFrames; ++frame) {
    memcpy(&g_model_input[frame * kCoefficients],
           &g_ring[((frame + g_ring_index) % kFrames) * kCoefficients], kCoefficients);
  }
  BB15Input input;
  input.data = g_model_input;
  input.type = akida::TensorType::uint8;
  input.dimensions = {1u, kFrames, kCoefficients, 1u};
  const BB15RunResult result = g_runner->infer(input);
  if (!result.ok() || result.type != akida::TensorType::int32 ||
      result.elementCount() < kClassCount) {
    return -1.0f;
  }
  // The engine's dequantization, then a two-class softmax.
  const int32_t* potentials = result.data<int32_t>();
  float logits[kClassCount];
  for (uint8_t i = 0u; i < kClassCount; ++i) {
    logits[i] = static_cast<float>(potentials[i] - g_output_shifts[i]) / g_output_scales[i];
  }
  ++g_inferences;
  return 1.0f / (1.0f + expf(logits[1 - kClapClass] - logits[kClapClass]));
}

void decide(float p) {
  g_smoothed = kAlpha * p + (1.0f - kAlpha) * g_smoothed;
  g_streak = g_smoothed >= kThreshold ? g_streak + 1u : 0u;
  if (g_print_scores) {
    Serial.print("S ");
    Serial.print(p, 3);
    Serial.print(' ');
    Serial.println(g_smoothed, 3);
  }
  if (g_streak < kStreak) return;
  ++g_detections;
  Serial.print("DOUBLECLAP score=");
  Serial.print(g_smoothed, 3);
  Serial.print(" ms=");
  Serial.println(millis());
  set_led(Led::Green);
  g_smoothed = 0.0f;
  g_streak = 0u;
  g_quiet_until_ms = millis() + kRefractoryMs;
}

void print_hex_line(char tag, const uint8_t* bytes, size_t count) {
  static const char kHex[] = "0123456789abcdef";
  static char line[2u + 2u * sizeof(g_raw_block) + 1u];
  size_t n = 0u;
  line[n++] = tag;
  line[n++] = ' ';
  for (size_t i = 0u; i < count; ++i) {
    line[n++] = kHex[bytes[i] >> 4];
    line[n++] = kHex[bytes[i] & 0x0Fu];
  }
  line[n++] = '\n';
  Serial.write(reinterpret_cast<const uint8_t*>(line), n);
}

void process_block() {
  extract_features();
  if (g_dump_left > 0u) {
    print_hex_line('P', reinterpret_cast<const uint8_t*>(g_raw_block), sizeof(g_raw_block));
    print_hex_line('F', g_block_features, sizeof(g_block_features));
    if (--g_dump_left == 0u) Serial.println("DUMP END");
  }
  memcpy(&g_mfcc_input[0], &g_mfcc_input[kBlockSamples], kHopSamples * sizeof(int16_t));
  g_block_fill = 0u;

  if (g_frames_seen < kFrames) return;
  if (static_cast<int32_t>(millis() - g_quiet_until_ms) < 0) return;
  set_led(Led::Off);
  const float p = infer();
  if (p < 0.0f) {
    Serial.println("[clap] infer failed");
    return;
  }
  decide(p);
}

void accumulate(const int16_t* raw, int16_t* filtered, size_t count) {
  size_t used = 0u;
  while (used < count) {
    const size_t take = min(count - used, kBlockSamples - g_block_fill);
    memcpy(&g_raw_block[g_block_fill], &raw[used], take * sizeof(int16_t));
    memcpy(&g_mfcc_input[kHopSamples + g_block_fill], &filtered[used], take * sizeof(int16_t));
    g_block_fill += take;
    used += take;
    if (g_block_fill == kBlockSamples) process_block();
  }
}

void on_pdm_data() {
  if (g_pdm_ready) ++g_pdm_overruns;
  g_pdm_ready = true;
}

void capture() {
  static int16_t raw[kPdmBufferSamples];
  const int bytes = PDM.read(g_pdm_samples, sizeof(g_pdm_samples));
  g_pdm_ready = false;
  const size_t count = static_cast<size_t>(bytes) / sizeof(int16_t);
  memcpy(raw, g_pdm_samples, count * sizeof(int16_t));
  remove_dc_offset(g_pdm_samples, count);
  accumulate(raw, g_pdm_samples, count);
}

// DUMP: restart the front end from a known state (DC blocker at zero, zero
// previous hop, block boundary now) so board_check.py can rebuild it exactly.
void start_dump() {
  g_dc_block = DcBlockState();
  memset(g_mfcc_input, 0, sizeof(g_mfcc_input));
  g_block_fill = 0u;
  g_frames_seen = 0u;
  g_dump_left = kDumpBlocks;
  Serial.print("DUMP START blocks=");
  Serial.println(kDumpBlocks);
}

void poll_commands() {
  static char line[16];
  static size_t n = 0u;
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c != '\n' && c != '\r') {
      if (n < sizeof(line) - 1u) line[n++] = c;
      continue;
    }
    line[n] = '\0';
    n = 0u;
    if (strcmp(line, "SCORES") == 0) {
      g_print_scores = !g_print_scores;
    } else if (strcmp(line, "DUMP") == 0) {
      start_dump();
    }
  }
}

void heartbeat() {
  if (static_cast<int32_t>(millis() - g_next_heartbeat_ms) < 0) return;
  g_next_heartbeat_ms = millis() + kHeartbeatMs;
  Serial.print("[clap] alive ms=");
  Serial.print(millis());
  Serial.print(" inferences=");
  Serial.print(g_inferences);
  Serial.print(" detections=");
  Serial.print(g_detections);
  Serial.print(" overruns=");
  Serial.println(g_pdm_overruns);
}

bool prepare_akida() {
  static BB15 bb15(g_pinout, g_config);
  static BB15Runner runner = bb15.createRunner();
  g_bb15 = &bb15;
  g_runner = &runner;
  if (bb15.begin() != BB15Status::Ok || runner.begin() != BB15Status::Ok) return false;
  g_model.setStorage(BB15ModelStorage::HostMemory);
  if (runner.loadModel(g_model) != BB15Status::Ok) return false;
  const akida::ProgramInfo info(program, static_cast<size_t>(program_len));
  if (!info.is_valid() || info.shifts().size < kClassCount || info.scales().size < kClassCount) {
    return false;
  }
  g_output_shifts = info.shifts().data;
  g_output_scales = info.scales().data;
  return true;
}

void fail(const char* stage) {
  Serial.print("[clap] result=FAIL stage=");
  Serial.println(stage);
  if (g_bb15 != nullptr && strcmp(stage, "akida") == 0) {
    Serial.print("[clap] detail=");
    g_bb15->printLastError(Serial);
  }
}

}  // namespace

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  set_led(Led::Blue);
  Serial.begin(115200);  // ignored on USB CDC
  const uint32_t start = millis();
  while (!Serial && millis() - start < kSerialWaitMs) {
  }
  Serial.println("[clap] clap_detector");
  if (!mfcc_begin()) return fail("mfcc");
  if (!prepare_akida()) return fail("akida");
  Serial.print("[clap] akida_ready model=");
  Serial.print(akida_model_path);
  Serial.print(" bytes=");
  Serial.print(static_cast<long>(akida_program_length_bytes));
  Serial.print(" threshold=");
  Serial.println(kThreshold, 2);
  PDM.onReceive(on_pdm_data);
  PDM.setBufferSize(kPdmBufferBytes);  // before begin(), which keeps the pointer
  PDM.setGain(kPdmGain);
  if (PDM.begin(1, kSampleRateHz) != 1) return fail("microphone");
  set_led(Led::Off);
  g_ready = true;
  Serial.println("[clap] listening");
}

void loop() {
  if (!g_ready) {  // setup failed: blink red, keep saying why
    set_led(Led::Red);
    delay(50);
    set_led(Led::Off);
    delay(950);
    Serial.println("[clap] result=FAIL see boot log");
    return;
  }
  poll_commands();
  heartbeat();
  if (g_pdm_ready) {
    capture();
  } else {
    delay(1);
  }
}

// Raw-audio recorder for the Nicla Vision. Streams the onboard PDM microphone
// over USB while the Mac asks for it (START / STOP lines) and shows REC or
// IDLE on the OLED. Protocol and usage: ../README.md.
//
// Microphone settings copied from the BB15 library's
// examples/bb15_nicla_vision_keyword_spotting (MIT), so recordings match what
// the keyword-spotting front end hears.

#include <Arduino.h>
#include <PDM.h>
#include <U8x8lib.h>

namespace {

constexpr uint32_t kSampleRateHz = 16000u;
// The DFSDM half-transfer hands the PDM library 256 samples at a time; 1024
// bytes is both the minimum and the required multiple of 512 bytes (KWS
// example). 512 samples = 32 ms per frame.
constexpr size_t kPdmBufferBytes = 1024u;
// Same gain as the KWS example. Close claps may clip: record.py reports it.
constexpr int kPdmGain = 12;

// Frame = magic (2 bytes) + overruns since START (uint16 LE) + PCM int16 LE.
constexpr uint8_t kFrameMagic[2] = {0xA5u, 0x5Au};
constexpr size_t kFrameHeaderBytes = 4u;
constexpr size_t kCommandMaxChars = 15u;
constexpr uint32_t kDisplayI2cClockHz = 400000u;

// ponytail: SSD1306 128x64 at 0x3C assumed (same guess as face-unlock's
// config.h); the carrier's controller is unconfirmed. If the screen stays dark,
// try U8X8_SH1106_128X64_NONAME_HW_I2C.
U8X8_SSD1306_128X64_NONAME_HW_I2C g_display(U8X8_PIN_NONE);

// Header and samples go out in one Serial.write: on native USB CDC every
// write is its own transfer.
alignas(4) uint8_t g_frame[kFrameHeaderBytes + kPdmBufferBytes];
// Written by the PDM interrupt, read by loop().
volatile bool g_pdm_ready = false;
volatile uint16_t g_pdm_overruns = 0u;

bool g_recording = false;
char g_command[kCommandMaxChars + 1];
size_t g_command_length = 0u;

// Interrupt context: only raise a flag. A buffer still unread when the next
// lands is lost audio, counted and reported to the Mac.
void on_pdm_data() {
  if (g_pdm_ready) {
    ++g_pdm_overruns;
  }
  g_pdm_ready = true;
}

// Fixed four-character width, so no clear (a full redraw takes ~25 ms of I2C,
// close to a 32 ms PDM buffer).
void show_state(const char* four_chars) {
  g_display.draw2x2String(4, 3, four_chars);
}

void start_recording() {
  show_state("REC ");
  digitalWrite(LEDR, LOW);  // Nicla LEDs are active low
  noInterrupts();
  g_pdm_overruns = 0u;
  g_pdm_ready = false;  // drop the buffer captured while idle
  interrupts();
  g_recording = true;
}

void stop_recording() {
  g_recording = false;
  digitalWrite(LEDR, HIGH);
  show_state("IDLE");
}

void handle_command() {
  if (strcmp(g_command, "START") == 0) {
    start_recording();
  } else if (strcmp(g_command, "STOP") == 0) {
    stop_recording();
  }
}

void read_commands() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n') {
      g_command[g_command_length] = '\0';
      handle_command();
      g_command_length = 0u;
    } else if (c != '\r' && g_command_length < kCommandMaxChars) {
      g_command[g_command_length++] = c;
    }
  }
}

// Always drains the PDM buffer so the overrun count means lost audio, but only
// sends it while recording.
void forward_microphone_buffer() {
  if (!g_pdm_ready) {
    return;
  }
  PDM.read(&g_frame[kFrameHeaderBytes], kPdmBufferBytes);
  g_pdm_ready = false;
  if (!g_recording) {
    return;
  }
  const uint16_t overruns = g_pdm_overruns;
  g_frame[2] = static_cast<uint8_t>(overruns & 0xFFu);
  g_frame[3] = static_cast<uint8_t>(overruns >> 8);
  Serial.write(g_frame, sizeof(g_frame));
}

// Buffer size first: begin() captures the buffer pointer (KWS example).
bool start_microphone() {
  PDM.onReceive(on_pdm_data);
  PDM.setBufferSize(kPdmBufferBytes);
  PDM.setGain(kPdmGain);
  return PDM.begin(1, kSampleRateHz) == 1;
}

}  // namespace

void setup() {
  pinMode(LEDR, OUTPUT);
  digitalWrite(LEDR, HIGH);
  Serial.begin(115200);  // baud is ignored on USB CDC
  g_frame[0] = kFrameMagic[0];
  g_frame[1] = kFrameMagic[1];

  g_display.setBusClock(kDisplayI2cClockHz);
  g_display.begin();
  g_display.setPowerSave(0);
  g_display.setFont(u8x8_font_chroma48medium8_r);
  g_display.clearDisplay();

  if (!start_microphone()) {
    show_state("MIC?");
    while (true) {
    }
  }
  show_state("IDLE");
}

void loop() {
  read_commands();
  if (g_recording && !Serial) {
    stop_recording();  // host closed the port without STOP
  }
  forward_microphone_buffer();
}

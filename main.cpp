/*
 * Resolution‑independent real‑time DFT/FFT visualiser (OpenGL 3.0)
 * ---------------------------------------------------------------------------
 *  • Window can be resized at runtime – geometry scales automatically.
 *  • Bottom half: magnitude spectrum (|DFT|) recomputed each frame (O(N²)) and FFT using Radix-2.
 *  • White guide lines:
 *      – Mid‑line (window / 2) separates waveform & spectrum.
 *      – Waveform zero reference at WAVE_ZERO_FRAC × height.
 */

#include <omp.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>   // for std::clamp

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

 // ---------------------------------------------------------------------------
 // ★★ USER‑TWEAKABLE GLOBAL CONSTANTS ★★
 // ---------------------------------------------------------------------------
 // Window initial size (can still be resized at runtime)
const int    INIT_WINDOW_W = 1600;
const int    INIT_WINDOW_H = 900;

// ---------------------------------------------------------------------------
// Dynamic window size (updated in framebuffer‑size callback)
// ---------------------------------------------------------------------------
static int gWinW = INIT_WINDOW_W;
static int gWinH = INIT_WINDOW_H;

// DFT settings
const int    N = 4096;      // samples / DFT points
const double SAMPLE_RATE_HZ = 4000.0;   // visible range (Hz)
const double MAX_SPECTRUM_FREQ_HZ = 8000.0;   // x‑axis limit for spectrum (≤ Nyquist)

// ─────────  signal selection  ─────────────────────────────────────────────
enum SignalType { SIG_SINE = 0, SIG_SQUARE, SIG_SAW, SIG_TRI };
SignalType  gSignal = SIG_SINE;
const char* SIGNAL_NAMES[4] = { "SIN", "SQR", "SAW", "TRI" };

// ───────── Wave-shape selector (lower-half button) ───────────────────────
enum WaveType { WAVE_SINE, WAVE_SQUARE, WAVE_TRIANGLE, WAVE_SAW };
WaveType    gWave = WAVE_SINE;

const float WBTN_W_FRAC = 0.12f;   // 12 % of width
const float WBTN_H_FRAC = 0.06f;   // 6 %  of height
const float WBTN_Y_FRAC = 0.4f;   // centred in lower half (¼ screen)
const float WBTN_MARGIN_FRAC = 0.75f;   // 25 % gap from right edge




// Waveform vertical placement (fractions of window height) ------------------
const double WAVE_ZERO_FRAC = 0.75;     // where 0‑V line sits (0 = bottom, 1 = top)
const double WAVE_AMPLITUDE_FRAC = 0.25;     // peak excursion as fraction of height

const double WAVE_DRAW_SCALE = 0.75;   // extra vertical gain for the sine
const double SPECTRUM_DRAW_SCALE = 0.75;   // extra vertical gain for the bars

// ── Slider (frequency control) ───────────────────────────────────────────
const double SLIDER_MIN_HZ = 1.0;
const double SLIDER_MAX_HZ = 1000.0;

const double MAGNITUDE = 0.75;
static double gSliderFreqHz = 50.0;   // initial position
static bool   gSliderDrag = false;    // true while knob held

// ───────── Slider geometry (all in window-fractions) ─────────────────────
const float SLIDER_X0_FRAC = 0.05f;   // track starts at  5 % of width
const float SLIDER_X1_FRAC = 0.45f;   // track ends   at 45 % of width
const float SLIDER_Y_FRAC = 0.95f;   // vertical position (0–1)
const float SLIDER_KNOB_R_FRAC = 0.015f;  // knob radius = 1.5 % of height


// Counters
double frameMs = 0.0;
static double gLastFrameTime = 0.0;   // used to compute Δt per frame

// ───────── Algorithm-toggle button (size as window-fractions) ────────────
bool        gUseFFT = false;  // false = DFT, true = FFT
const float BTN_W_FRAC = 0.12f;  // 12 % of window width
const float BTN_H_FRAC = 0.06f;  // 6 % of window height
const float BTN_Y_FRAC = SLIDER_Y_FRAC;   // same vertical centre as slider
const float BTN_MARGIN_FRAC = 0.02f;  // 2 % right-side margin

// ───────── Spectrum-scale toggle button ─────────────────────────────────
bool        gLogScale = false;        // false = linear, true = dB
const float LBTN_Y_FRAC = WBTN_Y_FRAC - 0.10f;   // one row below wave-button


// ---------------------------------------------------------------------------
// Signal + DFT buffers
// ---------------------------------------------------------------------------
static double in[N];
static double re[N];
static double im[N];
static std::vector<float> magnitude;            // |X[k]| (first N/2 bins)

// ---------------------------------------------------------------------------
// GLFW framebuffer resize callback – keeps projection & viewport in sync
// ---------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow*, int width, int height)
{
    gWinW = width;
    gWinH = height == 0 ? 1 : height;           // avoid div‑by‑zero

    glViewport(0, 0, gWinW, gWinH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, gWinW, 0, gWinH, -1, 1);
    glMatrixMode(GL_MODELVIEW);
}

// Mouse-button callback: start / stop dragging
void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    double mx, my;
    glfwGetCursorPos(win, &mx, &my);

    float x0 = gWinW * SLIDER_X0_FRAC;
    float x1 = gWinW * SLIDER_X1_FRAC;
    float y0 = gWinH * SLIDER_Y_FRAC;

    float  mxGL = static_cast<float>(mx);
    float  myGL = gWinH - static_cast<float>(my);

    float btnW = gWinW * BTN_W_FRAC;
    float btnH = gWinH * BTN_H_FRAC;
    float btnX0 = gWinW - btnW - gWinW * 0.02f;
    float btnX1 = btnX0 + btnW;
    float btnY0 = gWinH * BTN_Y_FRAC - btnH * 0.5f;
    float btnY1 = btnY0 + btnH;


    if (action == GLFW_PRESS) {
        float knobX = x0 + float((gSliderFreqHz - SLIDER_MIN_HZ) /
            (SLIDER_MAX_HZ - SLIDER_MIN_HZ)) * (x1 - x0);
        float knobR = gWinH * SLIDER_KNOB_R_FRAC;

        if (std::fabs(mxGL - knobX) < knobR && std::fabs(myGL - y0) < knobR)
            gSliderDrag = true;                 // grabbed knob
        else if (mxGL >= x0 && mxGL <= x1 && std::fabs(myGL - y0) < 5.0f)
            gSliderDrag = true;                 // clicked track

        if (gSliderDrag) {
            float t = std::clamp((mxGL - x0) / (x1 - x0), 0.0f, 1.0f);
            gSliderFreqHz = SLIDER_MIN_HZ + t * (SLIDER_MAX_HZ - SLIDER_MIN_HZ);
        }
    }
    else if (action == GLFW_RELEASE) {
        gSliderDrag = false;
    }

    // Button algorithm switch
    if (action == GLFW_PRESS &&
        mxGL >= btnX0 && mxGL <= btnX1 &&
        myGL >= btnY0 && myGL <= btnY1)
    {
        gUseFFT = !gUseFFT;        // flip between algorithms
    }

    // ----- wave-button hit-test ---------------------------------------------
    float wbW = gWinW * BTN_W_FRAC;
    float wbH = gWinH * WBTN_H_FRAC;
    float wbX0 = gWinW - wbW - gWinW * BTN_MARGIN_FRAC; 
    float wbX1 = wbX0 + wbW;
    float wbY0 = gWinH * WBTN_Y_FRAC - wbH * 0.5f;
    float wbY1 = wbY0 + wbH;

    if (action == GLFW_PRESS &&
        mxGL >= wbX0 && mxGL <= wbX1 &&
        myGL >= wbY0 && myGL <= wbY1)
    {
        gWave = static_cast<WaveType>((gWave + 1) % 4);   // cycle shapes
    }

    // ----- log/lin button hit-test -----------------------------------------
    float lbW = gWinW * BTN_W_FRAC;
    float lbH = gWinH * WBTN_H_FRAC;
    float lbX0 = gWinW - lbW - gWinW * BTN_MARGIN_FRAC;
    float lbX1 = lbX0 + lbW;
    float lbY0 = gWinH * LBTN_Y_FRAC - lbH * 0.5f;
    float lbY1 = lbY0 + lbH;

    if (action == GLFW_PRESS &&
        mxGL >= lbX0 && mxGL <= lbX1 &&
        myGL >= lbY0 && myGL <= lbY1)
    {
        gLogScale = !gLogScale;            // toggle scale
    }


}

// Cursor-motion callback: update while dragging
void cursor_position_callback(GLFWwindow*, double xpos, double ypos)
{
    if (!gSliderDrag) return;

    float x0 = gWinW * SLIDER_X0_FRAC;
    float x1 = gWinW * SLIDER_X1_FRAC;

    float mxGL = static_cast<float>(xpos);

    float t = std::clamp((mxGL - x0) / (x1 - x0), 0.0f, 1.0f);
    gSliderFreqHz = SLIDER_MIN_HZ + t * (SLIDER_MAX_HZ - SLIDER_MIN_HZ);
}


// ---------------------------------------------------------------------------
// Plain O(N²) DFT with optimisations
// ---------------------------------------------------------------------------
void dft(const double* x, double* outRe, double* outIm, int len)
{
    // ---- 1. Build / reuse LUT ------------------------------------------------
    static std::vector<double> cosLUT, sinLUT;
    static int cachedLen = 0;

    if (cachedLen != len) {
        cachedLen = len;
        cosLUT.resize(len);
        sinLUT.resize(len);

        const double twoPiOverN = 2.0 * M_PI / static_cast<double>(len);
        for (int n = 0; n < len; ++n) {
            double angle = twoPiOverN * n;
            cosLUT[n] = std::cos(angle);
            sinLUT[n] = std::sin(angle);
        }
    }

    #pragma omp parallel for
    // ---- 2. Outer loop over output bins k ------------------------------------
    for (int k = 0; k < len; ++k) {
        double sumRe = 0.0;
        double sumIm = 0.0;

        // Initial complex exponential e^{-j·0} = 1 + j0
        double c = 1.0;     // cos(0)
        double s = 0.0;     // sin(0)

        // Rotation increment e^{-j·2πk/N}
        double cosDelta = cosLUT[k];
        double sinDelta = -sinLUT[k];   // minus sign for e^{-jθ}

        // ---- 3. Inner loop ---------------------------------------------------
        for (int n = 0; n < len; ++n) {
            // x[n] * (c + j s)
            sumRe += x[n] * c;
            sumIm += x[n] * s;

            // Rotate (c,s) by Δ:  (c',s') = (c·cosΔ - s·sinΔ ,  c·sinΔ + s·cosΔ)
            double tmp = c * cosDelta - s * sinDelta;
            s = c * sinDelta + s * cosDelta;
            c = tmp;
        }

        outRe[k] = sumRe;
        outIm[k] = sumIm;
    }
}

void fft_radix2(const double* in, double* re, double* im, int len)
{
    /* ---- 0. Preconditions ------------------------------------------------ */
    if (len < 2 || (len & (len - 1)))           // not power-of-two
        return;

    /* ---- 1. Copy real input into working buffers ------------------------- */
    #pragma omp parallel for
    for (int n = 0; n < len; ++n) {
        re[n] = in[n];
        im[n] = 0.0;
    }

    /* ---- 2. Bit-reversal permutation ------------------------------------- */
    int j = 0;
    for (int i = 1; i < len - 1; ++i) {
        int bit = len >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    /* ---- 3. Twiddle-factor LUT  (negative sign for forward FFT) ---------- */
    static std::vector<double> wRe, wIm;
    static int cachedLen = 0;

    if (cachedLen != len) {
        cachedLen = len;
        wRe.resize(len / 2);
        wIm.resize(len / 2);
        const double twoPiOverN = -2.0 * M_PI / static_cast<double>(len);
        for (int k = 0; k < len / 2; ++k) {
            double angle = twoPiOverN * k;
            wRe[k] = std::cos(angle);
            wIm[k] = std::sin(angle);
        }
    }

    /* ---- 4. Iterative Danielson-Lanczos ---------------------------------- */
    for (int step = 1; step < len; step <<= 1) {
        int span = step << 1;                 // 2*step
        int twStep = len / span;                // twiddle stride

        // Parallelise each column (m) for large transforms
        #pragma omp parallel for if(len >= 1024)
        for (int m = 0; m < step; ++m) {
            double wr = wRe[m * twStep];
            double wi = wIm[m * twStep];

            for (int k = m; k < len; k += span) {
                int j = k + step;
                double tr = wr * re[j] - wi * im[j];
                double ti = wr * im[j] + wi * re[j];

                re[j] = re[k] - tr;
                im[j] = im[k] - ti;
                re[k] = re[k] + tr;
                im[k] = im[k] + ti;
            }
        }
    }
}



// ---------------------------------------------------------------------------
// Generate signal ‑ fills global "in[]" with one buffer of samples
// ---------------------------------------------------------------------------
inline void generateSignal(double cycles)
{
    const double twoPiOverN = 2.0 * M_PI / N;
    for (int n = 0; n < N; ++n) {
        double phase = twoPiOverN * cycles * n;           // 0‒2π
        double v;

        switch (gWave) {
        case WAVE_SINE:
            v = std::sin(phase);
            break;
        case WAVE_SQUARE:
            v = (std::sin(phase) >= 0.0) ? 1.0 : -1.0;
            break;
        case WAVE_TRIANGLE: {
            double t = phase / (2.0 * M_PI);              // 0‒1
            v = 4.0 * std::fabs(t - std::floor(t + 0.5)) - 1.0;
            break;
        }
        case WAVE_SAW:
        default: {
            double t = phase / (2.0 * M_PI);              // 0‒1
            v = 2.0 * (t - std::floor(t + 0.5));
            break;
        }
        }

        in[n] = MAGNITUDE * v;                            // uses your renamed constant
    }
}


// ---------------------------------------------------------------------------
// Compute (and normalise) magnitude spectrum
// ---------------------------------------------------------------------------
void computeMagnitude()
{
    magnitude.resize(N / 2);
    float maxMag = 0.0f;
    for (int k = 0; k < N / 2; ++k) {
        magnitude[k] = static_cast<float>(sqrt(re[k] * re[k] + im[k] * im[k]));
        if (magnitude[k] > maxMag) maxMag = magnitude[k];
    }
    if (maxMag > 0.0f) {
        for (float& m : magnitude) m /= maxMag;
    }
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
inline void drawHorizontalLine(float y, float r, float g, float b)
{
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex2f(0.0f, y);
    glVertex2f(static_cast<float>(gWinW), y);
    glEnd();
}

// ── Draw text ────────────────────────────────
inline void drawScaledStroke(const char* txt, float x, float y,
    float targetPx)        // desired glyph height in px
{
    const float GLUT_UNITS_PER_EM = 119.05f;        // for GLUT_STROKE_ROMAN
    float s = targetPx / GLUT_UNITS_PER_EM;         // scale factor

    glPushMatrix();
    glTranslatef(x, y, 0);
    glScalef(s, s, 1.0f);                           // uniform scale
    glColor3f(1, 1, 1);
    while (*txt) glutStrokeCharacter(GLUT_STROKE_ROMAN, *txt++);
    glPopMatrix();
}




void drawWaveform()
{
    const float waveZeroY = gWinH * static_cast<float>(WAVE_ZERO_FRAC);
    const float waveAmplitudeY = gWinH * static_cast<float>(WAVE_AMPLITUDE_FRAC * WAVE_DRAW_SCALE);


    // Zero reference line (white)
    drawHorizontalLine(waveZeroY, 1.0f, 1.0f, 1.0f);

    glColor3f(0.2f, 0.8f, 0.2f);
    glBegin(GL_LINE_STRIP);
    for (int n = 0; n < N; ++n) {
        float x = static_cast<float>(n) / (N - 1) * gWinW;
        float y = waveZeroY + static_cast<float>(in[n]) * waveAmplitudeY;
        glVertex2f(x, y);
    }
    glEnd();
}

void drawSpectrum()
{
    int binsToDraw = static_cast<int>(MAX_SPECTRUM_FREQ_HZ * N / SAMPLE_RATE_HZ);
    if (binsToDraw > N / 2) binsToDraw = N / 2;

    const float barWidth = static_cast<float>(gWinW) / binsToDraw;
    const float halfH = gWinH * 0.5f;


    glColor3f(0.2f, 0.4f, 1.0f);
    glBegin(GL_QUADS);
    for (int k = 0; k < binsToDraw; ++k) {
        
        float magVal = magnitude[k];
        float x0 = k * barWidth;
        float x1 = x0 + barWidth * 0.9f;
        float y0 = 0.0f;

        if (gLogScale) {                       // convert to 0-to-1 dB
            double db = 20.0 * std::log10(magVal + 1e-6);   // -∞ .. 0 dB
            magVal = static_cast<float>(
                std::clamp((db + 60.0) / 60.0, 0.0, 1.0)); // -60 dB floor
        }
        float y1 = magVal * halfH * static_cast<float>(SPECTRUM_DRAW_SCALE);
        glVertex2f(x0, y0);
        glVertex2f(x1, y0);
        glVertex2f(x1, y1);
        glVertex2f(x0, y1);
    }
    glEnd();
}

void drawSlider()
{
    float x0 = gWinW * SLIDER_X0_FRAC;
    float x1 = gWinW * SLIDER_X1_FRAC;
    float y = gWinH * SLIDER_Y_FRAC;
    float knobR = gWinH * SLIDER_KNOB_R_FRAC;

    // Track
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);  glVertex2f(x0, y);  glVertex2f(x1, y);  glEnd();

    // Knob
    float t = float((gSliderFreqHz - SLIDER_MIN_HZ) /
        (SLIDER_MAX_HZ - SLIDER_MIN_HZ));
    float kx = x0 + t * (x1 - x0);

    glColor3f(1.0f, 0.5f, 0.0f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(kx, y);
    for (int i = 0; i <= 20; ++i) {
        float a = i * 2.0f * float(M_PI) / 20.0f;
        glVertex2f(kx + std::cos(a) * knobR, y + std::sin(a) * knobR);
    }
    glEnd(); // Finish drawing slider

    float fontPx = knobR * 1.4f;                    // 40 % of knob diam.
    drawScaledStroke("Freq (Hz)", x0, y + knobR * 1.6f, fontPx);
}

void drawAlgoButton()
{
    float btnW = gWinW * BTN_W_FRAC;
    float btnH = gWinH * BTN_H_FRAC;          // ← dynamic
    float btnX0 = gWinW - btnW - gWinW * 0.02f;
    float btnX1 = btnX0 + btnW;
    float btnY0 = gWinH * BTN_Y_FRAC - btnH * 0.5f;
    float btnY1 = btnY0 + btnH;


    // Fill colour: green = FFT, red = DFT
    if (gUseFFT) glColor3f(0.1f, 0.7f, 0.2f);
    else         glColor3f(0.7f, 0.2f, 0.2f);

    glBegin(GL_QUADS);
    glVertex2f(btnX0, btnY0);
    glVertex2f(btnX1, btnY0);
    glVertex2f(btnX1, btnY1);
    glVertex2f(btnX0, btnY1);
    
    glEnd(); // Finish drawing rectangle/button move to text


    // ---------- centred "FFT"/"DFT" label (50 % of button height) ----------
    float fontPx = btnH * 0.50f;                    // 50 % of button
    drawScaledStroke(gUseFFT ? "FFT" : "DFT",
        btnX0 + btnW * 0.32f,              // x anchor
        btnY0 + btnH * 0.30f,          // y anchor
        fontPx);

    // ---------- frame-time text to the left of the button ------------------
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f ms", frameMs);

    drawScaledStroke(buf,
        btnX0 - gWinW * 0.12f,
        btnY0 + btnH * 0.30f,
        fontPx);
    
    // ---------- Frametime text -----------
    drawScaledStroke("Frametime:",
        btnX0 - gWinW * 0.25f,
        btnY0 + btnH * 0.30f,
        fontPx);


}

void drawWaveButton()
{
    float wbW = gWinW * BTN_W_FRAC;
    float wbH = gWinH * WBTN_H_FRAC;
    float wbX0 = gWinW - wbW - gWinW * BTN_MARGIN_FRAC;
    float wbX1 = wbX0 + wbW;
    float wbY0 = gWinH * WBTN_Y_FRAC - wbH * 0.5f;
    float wbY1 = wbY0 + wbH;


    // different fill colour per waveform
    switch (gWave) {
    case WAVE_SINE:     glColor3f(0.1f, 0.6f, 1.0f); break; // blue
    case WAVE_SQUARE:   glColor3f(1.0f, 0.4f, 0.1f); break; // orange
    case WAVE_TRIANGLE: glColor3f(0.2f, 0.8f, 0.2f); break; // green
    case WAVE_SAW:      glColor3f(0.8f, 0.1f, 0.6f); break; // magenta
    }

    glBegin(GL_QUADS);
    glVertex2f(wbX0, wbY0); glVertex2f(wbX1, wbY0);
    glVertex2f(wbX1, wbY1); glVertex2f(wbX0, wbY1);
    glEnd();

    const char* label[] = { "SINE", "SQUARE", "TRIANGLE", "SAW" };
    float fontPx = wbH * 0.50f;
    drawScaledStroke(label[gWave],
        wbX0 + wbW * 0.1f,
        wbY0 + wbH * 0.28f,
        fontPx);
}

void drawScaleButton()
{
    float lbW = gWinW * BTN_W_FRAC;
    float lbH = gWinH * WBTN_H_FRAC;
    float lbX0 = gWinW - lbW - gWinW * BTN_MARGIN_FRAC;
    float lbX1 = lbX0 + lbW;
    float lbY0 = gWinH * LBTN_Y_FRAC - lbH * 0.5f;
    float lbY1 = lbY0 + lbH;

    glColor3f(0.15f, 0.15f, 0.15f);    // grey background
    glBegin(GL_QUADS);
    glVertex2f(lbX0, lbY0); glVertex2f(lbX1, lbY0);
    glVertex2f(lbX1, lbY1); glVertex2f(lbX0, lbY1);
    glEnd();

    const char* txt = gLogScale ? "LOG" : "LIN";
    float fontPx = lbH * 0.50f;
    drawScaledStroke(txt,
        lbX0 + lbW * 0.25f,
        lbY0 + lbH * 0.28f,
        fontPx);
}



int main(int argc, char** argv)
{
    // OpenMP optimisation (25% of the threads)
    int half = omp_get_num_procs() / 4;
    omp_set_num_threads(half);
    printf("Using %d OpenMP threads\n", omp_get_max_threads());

    // -- Init GLUT --
    glutInit(&argc, argv);

    // ----- Initialise GLFW --------------------------------------------------
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* win = glfwCreateWindow(INIT_WINDOW_W, INIT_WINDOW_H,
        "DFT Demo", nullptr, nullptr);
    if (!win) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(0);                    // enable V-sync

    // ----- Viewport & projection -------------------------------------------
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);
    framebuffer_size_callback(win, INIT_WINDOW_W, INIT_WINDOW_H); // set up once

    // ----- Initialise GLEW --------------------------------------------------
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialise GLEW\n";
        return -1;
    }
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetCursorPosCallback(win, cursor_position_callback);

    gLastFrameTime = glfwGetTime();       // seed the frame-timer


    // ----- Main loop --------------------------------------------------------
    while (!glfwWindowShouldClose(win)) {
        // 1) Compute current sweep frequency and cycles-per-buffer
        double freqHz = gSliderFreqHz;
        double cycles = freqHz * N / SAMPLE_RATE_HZ;

        // 2) Generate signal and run DFT
        generateSignal(cycles);
        
        // Pick algorithm
        if (gUseFFT)
            fft_radix2(in, re, im, N);
        else
            dft(in, re, im, N);
        computeMagnitude();

        // 3) Update window title (shows frequency in Hz)
        // ------------------------------------------------------------------
        // Frame time timer
        double nowT = glfwGetTime();
        frameMs = (nowT - gLastFrameTime) * 1000.0;  // frametime in ms
        gLastFrameTime = nowT;

        // ------------------------------------------------------------------
        // Title
        std::ostringstream os;
        os << std::fixed << std::setprecision(1)
            << "Fourier transform Demo | f=" << freqHz << " Hz";

        glfwSetWindowTitle(win, os.str().c_str());


        // 4) Render
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Divider line (middle of window)
        drawHorizontalLine(gWinH * 0.5f, 1.0f, 1.0f, 1.0f);

        glLoadIdentity();
        drawWaveform();
        drawSpectrum();

        drawWaveform();
        drawSpectrum();
        drawSlider();       // draw slider and its text

        drawWaveform();
        drawSpectrum();
        drawSlider();
        drawAlgoButton();   // draw button, its text and frametime

        drawWaveButton();   // draw button for wave and scale change
        drawScaleButton();


        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

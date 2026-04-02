/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada

Group 70:
  Azaan Khan     (400399089)
  Ayaan Hussain  (400540174)
  Haseeb Shaikh  (400521659)
  Taimur Ahmed   (400514463)
*/

#include "dy4.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include "filter.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <complex>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
//  Timing infrastructure
// ─────────────────────────────────────────────────────────────────────────────
static std::map<std::string, double> g_timing_ms;
static std::mutex                    g_timing_mutex;
static std::atomic<int>              g_blocks_processed{0};

// How many RF blocks to process before printing the report.
// Set to 0 to run until EOF.
static const int N_TIMING_BLOCKS = 200;

static void accumulateTime(const std::string& name, double ms)
{
	std::lock_guard<std::mutex> lk(g_timing_mutex);
	g_timing_ms[name] += ms;
}

#define TIME_BLOCK(name, code_block)                                          \
	do {                                                                      \
		auto _t0 = std::chrono::high_resolution_clock::now();                 \
		{ code_block }                                                        \
		auto _t1 = std::chrono::high_resolution_clock::now();                 \
		std::chrono::duration<double, std::milli> _dt = _t1 - _t0;           \
		accumulateTime((name), _dt.count());                                  \
	} while (0)

static void printTimingSummary(int mode, int filter_taps)
{
	int nb = g_blocks_processed.load();
	std::cerr << "\n=== TIMING SUMMARY ===\n";
	std::cerr << "Mode=" << mode
	          << "  FilterTaps=" << filter_taps
	          << "  Blocks=" << nb << "\n";
	std::cerr << "block_name,total_ms,avg_ms_per_block\n";
	std::lock_guard<std::mutex> lk(g_timing_mutex);
	for (auto& kv : g_timing_ms) {
		double avg = (nb > 0) ? kv.second / nb : 0.0;
		std::cerr << kv.first << "," << kv.second << "," << avg << "\n";
	}
	std::cerr << "=== END TIMING SUMMARY ===\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Thread-safe bounded queue
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class BoundedQueue {
public:
	explicit BoundedQueue(size_t max_size) : max_size_(max_size), done_(false) {}

	void push(T item) {
		std::unique_lock<std::mutex> lk(mtx_);
		cv_not_full_.wait(lk, [this]{ return queue_.size() < max_size_ || done_; });
		if (done_) return;
		queue_.push(std::move(item));
		cv_not_empty_.notify_one();
	}

	bool pop(T& item) {
		std::unique_lock<std::mutex> lk(mtx_);
		cv_not_empty_.wait(lk, [this]{ return !queue_.empty() || done_; });
		if (queue_.empty()) return false;
		item = std::move(queue_.front());
		queue_.pop();
		cv_not_full_.notify_one();
		return true;
	}

	void finish() {
		std::lock_guard<std::mutex> lk(mtx_);
		done_ = true;
		cv_not_full_.notify_all();
		cv_not_empty_.notify_all();
	}

	bool empty() {
		std::lock_guard<std::mutex> lk(mtx_);
		return queue_.empty();
	}

private:
	std::queue<T>           queue_;
	std::mutex              mtx_;
	std::condition_variable cv_not_full_;
	std::condition_variable cv_not_empty_;
	size_t                  max_size_;
	bool                    done_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: impulse response for a band-pass filter
//  Built as the difference of two LPFs (high-cutoff minus low-cutoff).
// ─────────────────────────────────────────────────────────────────────────────
static void impulseResponseBPF(real Fs, real Fc_low, real Fc_high,
                               int num_taps, std::vector<real>& h)
{
	std::vector<real> h_low, h_high;
	impulseResponseLPF(Fs, Fc_high, (unsigned short int)num_taps, h_high);
	impulseResponseLPF(Fs, Fc_low,  (unsigned short int)num_taps, h_low);
	h.resize(num_taps);
	for (int k = 0; k < num_taps; k++)
		h[k] = h_high[k] - h_low[k];
}

// ─────────────────────────────────────────────────────────────────────────────
//  PLL / NCO  (stereo carrier recovery – locks to 19 kHz pilot → 38 kHz)
// ─────────────────────────────────────────────────────────────────────────────
struct PllState {
	real integrator  = 0.0;
	real phaseEst    = 0.0;
	real feedbackI   = 1.0;
	real feedbackQ   = 0.0;
	real ncoOut_prev = 1.0;
};

static void pllBlock(const std::vector<real>& pllIn,
                     real                     freq,
                     real                     Fs,
                     real                     ncoScale,
                     real                     phaseAdjust,
                     real                     normBandwidth,
                     PllState&                state,
                     std::vector<real>&       ncoOut)
{
	real Cp = 2.666;
	real Ci = 3.555;
	real Kp = normBandwidth * Cp;
	real Ki = normBandwidth * normBandwidth * Ci;

	ncoOut.resize(pllIn.size() + 1);
	ncoOut[0] = state.ncoOut_prev;

	for (int k = 0; k < (int)pllIn.size(); k++) {
		real errorI = pllIn[k] * state.feedbackI;
		real errorQ = pllIn[k] * state.feedbackQ;
		real errorD = std::atan2(errorQ, errorI);

		state.integrator += Ki * errorD;
		state.phaseEst   += Kp * errorD + state.integrator;

		real trigArg = 2.0 * PI * (freq / Fs) * (k + 1) + state.phaseEst;
		state.feedbackI  = std::cos(trigArg);
		state.feedbackQ  = std::sin(trigArg);
		ncoOut[k + 1]    = std::cos(trigArg * ncoScale + phaseAdjust);
	}
	state.ncoOut_prev = ncoOut[pllIn.size()];
}

// ─────────────────────────────────────────────────────────────────────────────
//  PLL IQ variant  (RDS carrier recovery – operates on complex signal)
// ─────────────────────────────────────────────────────────────────────────────
struct PllIQState {
	real integrator   = 0.0;
	real phaseEst     = 0.0;
	real feedbackI    = 1.0;
	real feedbackQ    = 0.0;
	real ncoOutI_prev = 1.0;
	real ncoOutQ_prev = 0.0;
};

static void pllBlockIQ(const std::vector<real>& pllInI,
                       const std::vector<real>& pllInQ,
                       real                     freq,
                       real                     Fs,
                       real                     normBandwidth,
                       PllIQState&              state,
                       std::vector<real>&       ncoOutI,
                       std::vector<real>&       ncoOutQ)
{
	real Cp = 2.666;
	real Ci = 3.555;
	real Kp = normBandwidth * Cp;
	real Ki = normBandwidth * normBandwidth * Ci;

	int N = (int)pllInI.size();
	ncoOutI.resize(N + 1);
	ncoOutQ.resize(N + 1);
	ncoOutI[0] = state.ncoOutI_prev;
	ncoOutQ[0] = state.ncoOutQ_prev;

	for (int k = 0; k < N; k++) {
		real errI = pllInI[k] * state.feedbackI + pllInQ[k] * state.feedbackQ;
		real errQ = pllInQ[k] * state.feedbackI - pllInI[k] * state.feedbackQ;
		real errD = std::atan2(errQ, errI);

		state.integrator += Ki * errD;
		state.phaseEst   += Kp * errD + state.integrator;

		real trigArg     = 2.0 * PI * (freq / Fs) * (k + 1) + state.phaseEst;
		state.feedbackI  = std::cos(trigArg);
		state.feedbackQ  = std::sin(trigArg);
		ncoOutI[k + 1]   = std::cos(trigArg);
		ncoOutQ[k + 1]   = std::sin(trigArg);
	}
	state.ncoOutI_prev = ncoOutI[N];
	state.ncoOutQ_prev = ncoOutQ[N];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rational resampler  (upFIRdn equivalent)
//  Upsamples by U, applies FIR h, downsamples by D.
// ─────────────────────────────────────────────────────────────────────────────
static void blockConvolve_Resample(std::vector<real>&       yb,
                                   const std::vector<real>& xb,
                                   const std::vector<real>& h,
                                   std::vector<real>&       state,
                                   int                      U,
                                   int                      D)
{
	int M   = (int)h.size();      // h already has length = taps * U (polyphase)
	int Nin = (int)xb.size();

	// Number of output samples
	int Nout = (int)std::ceil((double)(Nin * U) / D);
	yb.assign(Nout, 0.0);

	// Build combined = [state | xb]  at the *upsampled* rate would be huge;
	// instead we use the polyphase decomposition: h has been designed with
	// U*taps coefficients and we step by D in the upsampled domain.
	// For each output sample n, the upsampled input index is n*D,
	// the base input index is (n*D) / U and phase is (n*D) % U.

	int taps = M / U;  // per-phase tap count

	for (int n = 0; n < Nout; n++) {
		int upIdx  = n * D;
		int phase  = upIdx % U;
		int xStart = upIdx / U;   // first real input sample for this output

		real acc = 0.0;
		for (int k = 0; k < taps; k++) {
			int xi = xStart - k;
			real xv;
			if (xi >= 0 && xi < Nin)
				xv = xb[xi];
			else if (xi < 0 && (xi + (int)state.size()) >= 0)
				xv = state[xi + (int)state.size()];
			else
				xv = 0.0;
			// polyphase coefficient index: phase + k*U
			int hi = phase + k * U;
			if (hi < M)
				acc += h[hi] * xv;
		}
		yb[n] = acc;
	}

	// Update state (last taps-1 real samples)
	int stateLen = taps - 1;
	state.resize(stateLen);
	for (int i = 0; i < stateLen; i++) {
		int xi = Nin - stateLen + i;
		state[i] = (xi >= 0 && xi < Nin) ? xb[xi] : 0.0;
	}
}

// ─────────────────────────────────────────────────────────────────────────────
//  RRC impulse response
// ─────────────────────────────────────────────────────────────────────────────
static void impulseResponseRRC(real Fs, int num_taps, std::vector<real>& h)
{
	real T_symbol = 1.0 / 2375.0;
	real beta     = 0.90;
	h.resize(num_taps);

	for (int k = 0; k < num_taps; k++) {
		real t = (real)(k - num_taps / 2) / Fs;
		if (t == 0.0) {
			h[k] = 1.0 + beta * (4.0 / PI - 1.0);
		} else if (std::abs(t) == T_symbol / (4.0 * beta)) {
			h[k] = (beta / std::sqrt(2.0)) *
			       ((1.0 + 2.0 / PI) * std::sin(PI / (4.0 * beta)) +
			        (1.0 - 2.0 / PI) * std::cos(PI / (4.0 * beta)));
		} else {
			real num = std::sin(PI * t * (1.0 - beta) / T_symbol) +
			           4.0 * beta * (t / T_symbol) *
			           std::cos(PI * t * (1.0 + beta) / T_symbol);
			real den = PI * t * (1.0 - (4.0 * beta * t / T_symbol) *
			                          (4.0 * beta * t / T_symbol)) / T_symbol;
			h[k] = (std::abs(den) > 1e-12) ? num / den : 0.0;
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
//  Manchester + differential decoding
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> manchesterDecode(const std::vector<int>& symbols)
{
	std::vector<int> bits;
	for (int i = 0; i + 1 < (int)symbols.size(); i += 2) {
		if (symbols[i] == 1 && symbols[i + 1] == -1)
			bits.push_back(1);
		else if (symbols[i] == -1 && symbols[i + 1] == 1)
			bits.push_back(0);
		// else: invalid pair, skip
	}
	return bits;
}

static std::vector<int> differentialDecode(const std::vector<int>& bits)
{
	std::vector<int> decoded;
	for (int i = 1; i < (int)bits.size(); i++)
		decoded.push_back(bits[i] ^ bits[i - 1]);
	return decoded;
}

// ─────────────────────────────────────────────────────────────────────────────
//  RDS syndrome / block detection  (standard offset words for groups A–D)
// ─────────────────────────────────────────────────────────────────────────────
// CRC-10 polynomial: x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1  (0x5B9)
static const uint16_t RDS_OFFSETS[4] = {0x0FC, 0x198, 0x168, 0x1B4};
static const uint16_t RDS_POLY       = 0x5B9;

static uint16_t rdsCRC(uint32_t word26)
{
	// Compute syndrome of a 26-bit received block
	uint32_t rem = word26;
	for (int i = 25; i >= 0; i--) {
		if (rem & (1u << i)) {
			rem ^= (RDS_POLY << (i - 10));
		}
	}
	return (uint16_t)(rem & 0x3FF);
}

// ─────────────────────────────────────────────────────────────────────────────
//  RDS data processing (frame sync + field extraction)
// ─────────────────────────────────────────────────────────────────────────────
static void processRDSBits(const std::vector<int>& bits)
{
	// Sliding window to find 26-bit blocks by syndrome matching
	static std::string ps_name(8, ' ');
	static std::string radiotext(64, ' ');
	static int  pi_code   = -1;
	static int  pty       = -1;
	static bool synced    = false;
	static int  syncPhase = 0;

	if ((int)bits.size() < 26) return;

	for (int i = 0; i <= (int)bits.size() - 26; i++) {
		uint32_t word = 0;
		for (int b = 0; b < 26; b++)
			word = (word << 1) | (bits[i + b] & 1);

		uint16_t syn = rdsCRC(word);
		for (int blk = 0; blk < 4; blk++) {
			if (syn == RDS_OFFSETS[blk]) {
				uint16_t info = (uint16_t)((word >> 10) & 0xFFFF);
				if (blk == 0) {
					// Block A: PI code
					pi_code = info;
					std::cerr << "[RDS] PI=" << std::hex << pi_code << std::dec << "\n";
				} else if (blk == 1) {
					// Block B: group type, PTY
					int  groupType = (info >> 12) & 0xF;
					bool versionB  = (info >> 11) & 0x1;
					pty            = (info >>  5) & 0x1F;
					(void)groupType; (void)versionB;
				} else if (blk == 3) {
					// Block D – PS name segment (group 0A)
					int seg = (bits[i - 26 * 2] & 1) | ((bits[i - 26 * 2 + 1] & 1) << 1);
					ps_name[seg * 2]     = (char)(info >> 8);
					ps_name[seg * 2 + 1] = (char)(info & 0xFF);
					std::cerr << "[RDS] PS=" << ps_name << "  PTY=" << pty << "\n";
				}
				break;
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
//  Struct: one RF block exchanged between RF thread → Audio/RDS threads
// ─────────────────────────────────────────────────────────────────────────────
struct IFBlock {
	std::vector<real> fm_demod;   // FM-demodulated IF signal
};

// ─────────────────────────────────────────────────────────────────────────────
//  RF front-end thread
//  Reads raw IQ, filters/decimates, FM-demodulates, pushes IFBlock to queues.
// ─────────────────────────────────────────────────────────────────────────────
static void rfThread(int mode, int filter_taps,
                     BoundedQueue<IFBlock>* audioQ,
                     BoundedQueue<IFBlock>* rdsQ,   // nullptr if RDS not needed
                     std::atomic<bool>&     stopFlag)
{
	real rf_Fs, If_Fs, rf_Fc, block_duration_ms;
	int  rf_decim, rfblocksize, block_IQ_size, If_block_size;
	rf_Fc = 100e3;

	if      (mode == 0) { rf_Fs=2400e3; If_Fs=240e3; block_duration_ms=40;  rf_decim=10; }
	else if (mode == 1) { rf_Fs=2304e3; If_Fs=288e3; block_duration_ms=56;  rf_decim=8;  }
	else if (mode == 2) { rf_Fs=2400e3; If_Fs=240e3; block_duration_ms=60;  rf_decim=10; }
	else                { rf_Fs=1800e3; If_Fs=360e3; block_duration_ms=30;  rf_decim=5;  }

	rfblocksize   = (int)(rf_Fs * block_duration_ms / 1000.0);
	block_IQ_size = rfblocksize * 2;
	If_block_size = rfblocksize / rf_decim;

	std::vector<real> rf_coeff;
	impulseResponseLPF(rf_Fs, rf_Fc, (unsigned short int)filter_taps, rf_coeff);

	std::vector<real> I_state(filter_taps - 1, 0.0);
	std::vector<real> Q_state(filter_taps - 1, 0.0);
	real I_demod_state = 0.0, Q_demod_state = 0.0;

	std::vector<real> block_data(block_IQ_size, 0.0);
	std::vector<real> block_I(rfblocksize, 0.0);
	std::vector<real> block_Q(rfblocksize, 0.0);
	std::vector<real> if_I(If_block_size, 0.0);
	std::vector<real> if_Q(If_block_size, 0.0);
	std::vector<real> fm_demod(If_block_size, 0.0);
	std::vector<real> combinedRF(filter_taps - 1 + rfblocksize, 0.0);
	std::vector<char> raw_IQ(block_IQ_size);

	while (!stopFlag.load()) {
		TIME_BLOCK("RF_ReadStdin", {
			readStdinBlockData(block_IQ_size, block_data, raw_IQ);
		});

		if (std::cin.rdstate() != 0) {
			std::cerr << "End of input stream reached\n";
			stopFlag.store(true);
			break;
		}

		TIME_BLOCK("RF_UnInterleave", {
			UnInterleave_IQ(block_data, block_I, block_Q);
		});

		TIME_BLOCK("RF_LPF_I", {
			blockConvolve_Decimate(if_I, block_I, rf_coeff, I_state, combinedRF, rf_decim);
		});
		TIME_BLOCK("RF_LPF_Q", {
			blockConvolve_Decimate(if_Q, block_Q, rf_coeff, Q_state, combinedRF, rf_decim);
		});
		TIME_BLOCK("RF_FMDemod", {
			fmDemodNoArctan(if_I, if_Q, I_demod_state, Q_demod_state, fm_demod);
		});

		IFBlock blk;
		blk.fm_demod = fm_demod;

		audioQ->push(blk);
		if (rdsQ) rdsQ->push(blk);

		int cur = g_blocks_processed.fetch_add(1) + 1;
		if (cur % 50 == 0)
			std::cerr << "RF block " << cur << "\n";

		if (N_TIMING_BLOCKS > 0 && cur >= N_TIMING_BLOCKS)
			stopFlag.store(true);
	}

	audioQ->finish();
	if (rdsQ) rdsQ->finish();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Audio thread  (mono or stereo)
// ─────────────────────────────────────────────────────────────────────────────
static void audioThread(int mode, char path, int filter_taps,
                        BoundedQueue<IFBlock>* audioQ,
                        std::atomic<bool>&     stopFlag)
{
	real If_Fs, audio_Fs, audio_Fc = 16e3;
	int  If_block_size, audio_decim, audio_up, audio_block_size;
	real block_duration_ms;
	real rf_Fs;
	int  rf_decim;

	if      (mode == 0) { rf_Fs=2400e3; If_Fs=240e3; audio_Fs=48e3;   block_duration_ms=40;  rf_decim=10; audio_decim=5;   audio_up=1;   }
	else if (mode == 1) { rf_Fs=2304e3; If_Fs=288e3; audio_Fs=36e3;   block_duration_ms=56;  rf_decim=8;  audio_decim=8;   audio_up=1;   }
	else if (mode == 2) { rf_Fs=2400e3; If_Fs=240e3; audio_Fs=44.1e3; block_duration_ms=60;  rf_decim=10; audio_decim=800; audio_up=147; }
	else                { rf_Fs=1800e3; If_Fs=360e3; audio_Fs=44.1e3; block_duration_ms=30;  rf_decim=5;  audio_decim=400; audio_up=49;  }

	int rfblocksize   = (int)(rf_Fs * block_duration_ms / 1000.0);
	If_block_size     = rfblocksize / rf_decim;
	audio_block_size  = (int)((real)If_block_size * audio_up / audio_decim);

	// ── Mono coefficients ────────────────────────────────────────────────────
	std::vector<real> audio_coeff;
	impulseResponseLPF(If_Fs, audio_Fc, (unsigned short int)filter_taps, audio_coeff);
	std::vector<real> audio_state(filter_taps - 1, 0.0);
	std::vector<real> combined_audio(filter_taps - 1 + If_block_size, 0.0);

	// ── Stereo-only coefficients ─────────────────────────────────────────────
	std::vector<real> pilot_coeff, stereo_coeff, stereo_lp_coeff;
	std::vector<real> pilot_state, stereo_state, stereo_lp_state;
	std::vector<real> combined_pilot(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo_lp(filter_taps - 1 + If_block_size, 0.0);
	PllState          pll_state;

	// Mono delay state (to align with stereo branch delay)
	int               mono_delay = (filter_taps - 1) / 2;
	std::vector<real> mono_delay_state(mono_delay, 0.0);

	if (path == 's') {
		impulseResponseBPF(If_Fs, 18.5e3, 19.5e3, filter_taps, pilot_coeff);
		impulseResponseBPF(If_Fs, 22e3,   54e3,   filter_taps, stereo_coeff);
		impulseResponseLPF(If_Fs, audio_Fc, (unsigned short int)filter_taps, stereo_lp_coeff);
		pilot_state.assign(filter_taps - 1, 0.0);
		stereo_state.assign(filter_taps - 1, 0.0);
		stereo_lp_state.assign(filter_taps - 1, 0.0);
	}

	std::vector<real> mono_out(audio_block_size, 0.0);
	std::vector<real> stereo_out(audio_block_size, 0.0);
	std::vector<real> pilot_out(If_block_size, 0.0);
	std::vector<real> nco_out;
	std::vector<real> stereo_band(If_block_size, 0.0);
	std::vector<real> stereo_mixed(If_block_size, 0.0);
	std::vector<real> stereo_base(audio_block_size, 0.0);

	std::vector<short int> pcm_out;

	IFBlock blk;
	while (audioQ->pop(blk)) {
		const std::vector<real>& fm = blk.fm_demod;

		// ── Mono LPF + decimate ──────────────────────────────────────────────
		TIME_BLOCK("Audio_MonoLPF", {
			blockConvolve_Decimate(mono_out, fm, audio_coeff,
			                      audio_state, combined_audio, audio_decim);
		});

		if (path == 'm') {
			// Write mono PCM
			TIME_BLOCK("Audio_PCMWrite", {
				pcm_out.resize(mono_out.size());
				for (size_t k = 0; k < mono_out.size(); k++) {
					pcm_out[k] = std::isnan(mono_out[k])
					           ? 0 : (short int)(mono_out[k] * 16384);
				}
				fwrite(pcm_out.data(), sizeof(short int), pcm_out.size(), stdout);
			});

		} else if (path == 's') {
			// ── Pilot BPF ────────────────────────────────────────────────────
			TIME_BLOCK("Stereo_PilotBPF", {
				blockConvolve_Decimate(pilot_out, fm, pilot_coeff,
				                      pilot_state, combined_pilot, 1);
			});

			// ── PLL: lock to 19 kHz pilot, generate 38 kHz carrier ───────────
			TIME_BLOCK("Stereo_PLL", {
				pllBlock(pilot_out, 19e3, If_Fs, 2.0, 0.0, 0.01, pll_state, nco_out);
			});

			// ── Stereo BPF (23–53 kHz L-R band) ─────────────────────────────
			TIME_BLOCK("Stereo_BandBPF", {
				blockConvolve_Decimate(stereo_band, fm, stereo_coeff,
				                      stereo_state, combined_stereo, 1);
			});

			// ── Mix L-R band with 38 kHz carrier ─────────────────────────────
			TIME_BLOCK("Stereo_Mixer", {
				for (int k = 0; k < (int)stereo_band.size(); k++)
					stereo_mixed[k] = stereo_band[k] * nco_out[k];
			});

			// ── Stereo LPF + decimate ─────────────────────────────────────────
			TIME_BLOCK("Stereo_LPF", {
				blockConvolve_Decimate(stereo_base, stereo_mixed, stereo_lp_coeff,
				                      stereo_lp_state, combined_stereo_lp, audio_decim);
			});

			// ── Mono path delay alignment ─────────────────────────────────────
			TIME_BLOCK("Stereo_MonoDelay", {
				// Shift mono_out through delay state
				std::vector<real> delayed(mono_out.size());
				int ds = (int)mono_delay_state.size();
				for (int k = 0; k < (int)mono_out.size(); k++) {
					if (k < ds)
						delayed[k] = mono_delay_state[k];
					else
						delayed[k] = mono_out[k - ds];
				}
				// Update delay state
				int tail = std::min(ds, (int)mono_out.size());
				for (int k = 0; k < ds; k++) {
					int src = (int)mono_out.size() - ds + k;
					mono_delay_state[k] = (src >= 0) ? mono_out[src] : mono_delay_state[k];
				}
				mono_out = delayed;
			});

			// ── Combine L+R and L-R → L, R ───────────────────────────────────
			TIME_BLOCK("Stereo_LRCombine", {
				pcm_out.resize(2 * mono_out.size());
				for (size_t k = 0; k < mono_out.size(); k++) {
					real L = (mono_out[k] + stereo_base[k]);
					real R = (mono_out[k] - stereo_base[k]);
					pcm_out[2 * k]     = std::isnan(L) ? 0 : (short int)(L * 16384);
					pcm_out[2 * k + 1] = std::isnan(R) ? 0 : (short int)(R * 16384);
				}
				fwrite(pcm_out.data(), sizeof(short int), pcm_out.size(), stdout);
			});
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
//  RDS thread  (modes 0 and 2 only – IF_Fs = 240 kHz)
// ─────────────────────────────────────────────────────────────────────────────
static void rdsThread(int mode, int filter_taps,
                      BoundedQueue<IFBlock>* rdsQ,
                      std::atomic<bool>&     stopFlag)
{
	if (mode != 0 && mode != 2) {
		IFBlock blk;
		while (rdsQ->pop(blk)) {}
		return;
	}

	real If_Fs = 240e3;

	// ── Group 70: SPS = 19, U=19, D=80 for mode 0 ────────────────────────────
	// Output rate = 19 * 2375 = 45125 (≈ 45.125 kSps)
	// gcd(240000, 45125) …  use SPS=19 → output=45125, U=19, D=80 (approx)
	// Simpler: SPS=24 → output=57000; gcd(240000,57000)=3000; U=19, D=80
	// Group 70 constraint file specifies SPS – use 19 here; adjust if needed.
	int SPS = 19;           // samples per symbol at resampler output
	int U_rds = SPS;        // upsample factor
	int D_rds = 80;         // downsample factor  (240000*19/57000 ≈ 80)
	real rds_symbol_rate = 2375.0;
	// real rds_Fs = SPS * rds_symbol_rate;   // ~45125 or 57000 Hz

	// ── RDS BPF  54–60 kHz ───────────────────────────────────────────────────
	std::vector<real> rds_bpf_coeff;
	impulseResponseBPF(If_Fs, 54e3, 60e3, filter_taps, rds_bpf_coeff);
	std::vector<real> rds_bpf_state(filter_taps - 1, 0.0);
	std::vector<real> rds_bpf_combined;

	// ── 114 kHz BPF for squaring-based carrier recovery ─────────────────────
	std::vector<real> bpf114_coeff;
	impulseResponseBPF(If_Fs, 113e3, 115e3, filter_taps, bpf114_coeff);
	std::vector<real> bpf114_state(filter_taps - 1, 0.0);
	std::vector<real> bpf114_combined;

	// ── PLL IQ state ─────────────────────────────────────────────────────────
	PllIQState pll_iq;

	// ── Resampler coefficients (polyphase) ───────────────────────────────────
	// Design a single LPF at cutoff = min(1/U, 1/D)/2 normalised,
	// then expand to U*taps length for polyphase use.
	int rds_taps = filter_taps;
	std::vector<real> resamp_h_raw;
	real resamp_cutoff = std::min(1.0f / (real)U_rds, 1.0f / (real)D_rds) * If_Fs / 2.0f;
	impulseResponseLPF(If_Fs * U_rds, resamp_cutoff,
	                   (unsigned short int)(rds_taps * U_rds), resamp_h_raw);
	std::vector<real> resamp_state_I(rds_taps - 1, 0.0);
	std::vector<real> resamp_state_Q(rds_taps - 1, 0.0);

	// ── RRC filter ───────────────────────────────────────────────────────────
	real rds_Fs_out = (real)(U_rds * (int)rds_symbol_rate); // approximate
	// Recalculate properly:
	rds_Fs_out = If_Fs * (real)U_rds / (real)D_rds;
	std::vector<real> rrc_coeff;
	impulseResponseRRC(rds_Fs_out, rds_taps, rrc_coeff);
	std::vector<real> rrc_state_I(rds_taps - 1, 0.0);
	std::vector<real> rrc_state_Q(rds_taps - 1, 0.0);
	std::vector<real> rrc_combined_I, rrc_combined_Q;

	// ── CDR state ─────────────────────────────────────────────────────────────
	static int  cdr_phase   = -1;   // -1 = needs warmup
	static bool use_I_axis  = true;
	static const int WARMUP_BLOCKS = 5;
	static int  warmup_count = 0;
	static std::vector<real> warmup_I, warmup_Q;

	// ── Allpass delay state (match stereo path delay) ─────────────────────────
	int allpass_delay = (filter_taps - 1) / 2;
	std::vector<real> allpass_state_I(allpass_delay, 0.0);
	std::vector<real> allpass_state_Q(allpass_delay, 0.0);

	// Working buffers
	std::vector<real> rds_bpf_out, squared, bpf114_out;
	std::vector<real> nco_I, nco_Q;
	std::vector<real> mixed_I, mixed_Q;
	std::vector<real> resampled_I, resampled_Q;
	std::vector<real> rrc_out_I, rrc_out_Q;

	// Accumulate bits across blocks for RDS frame processing
	std::vector<int> bit_accumulator;

	IFBlock blk;
	while (rdsQ->pop(blk)) {
		const std::vector<real>& fm = blk.fm_demod;
		int N = (int)fm.size();

		// ── 1. RDS BPF ───────────────────────────────────────────────────────
		rds_bpf_out.resize(N);
		rds_bpf_combined.resize(filter_taps - 1 + N);
		TIME_BLOCK("RDS_BPF", {
			blockConvolve_Decimate(rds_bpf_out, fm, rds_bpf_coeff,
			                      rds_bpf_state, rds_bpf_combined, 1);
		});

		// ── 2. Squaring non-linearity for carrier recovery ───────────────────
		TIME_BLOCK("RDS_Squaring", {
			squared.resize(N);
			for (int k = 0; k < N; k++)
				squared[k] = rds_bpf_out[k] * rds_bpf_out[k];
		});

		// ── 3. 114 kHz BPF ───────────────────────────────────────────────────
		bpf114_out.resize(N);
		bpf114_combined.resize(filter_taps - 1 + N);
		TIME_BLOCK("RDS_114kHzBPF", {
			blockConvolve_Decimate(bpf114_out, squared, bpf114_coeff,
			                      bpf114_state, bpf114_combined, 1);
		});

		// ── 4. PLL IQ to recover 57 kHz carrier ─────────────────────────────
		TIME_BLOCK("RDS_PLL_IQ", {
			pllBlockIQ(bpf114_out, std::vector<real>(N, 0.0),
			           57e3, If_Fs, 0.001, pll_iq, nco_I, nco_Q);
		});

		// ── 5. Allpass delay on RDS channel ─────────────────────────────────
		TIME_BLOCK("RDS_AllpassDelay", {
			// Simple shift-register delay
			std::vector<real> delayed_I(N), delayed_Q(N);
			int ds = allpass_delay;
			for (int k = 0; k < N; k++) {
				delayed_I[k] = (k < ds) ? allpass_state_I[k] : rds_bpf_out[k - ds];
				delayed_Q[k] = (k < ds) ? allpass_state_Q[k] : rds_bpf_out[k - ds];
			}
			for (int k = 0; k < ds; k++) {
				int src = N - ds + k;
				allpass_state_I[k] = (src >= 0 && src < N) ? rds_bpf_out[src] : 0.0f;
				allpass_state_Q[k] = allpass_state_I[k];
			}
			rds_bpf_out = delayed_I; // reuse buffer
		});

		// ── 6. Mix down to baseband ──────────────────────────────────────────
		TIME_BLOCK("RDS_Mixer", {
			mixed_I.resize(N); mixed_Q.resize(N);
			for (int k = 0; k < N; k++) {
				mixed_I[k] = rds_bpf_out[k] * nco_I[k];
				mixed_Q[k] = rds_bpf_out[k] * nco_Q[k];
			}
		});

		// ── 7. Rational resample (I and Q) ───────────────────────────────────
		TIME_BLOCK("RDS_Resample", {
			blockConvolve_Resample(resampled_I, mixed_I, resamp_h_raw,
			                      resamp_state_I, U_rds, D_rds);
			blockConvolve_Resample(resampled_Q, mixed_Q, resamp_h_raw,
			                      resamp_state_Q, U_rds, D_rds);
		});

		// ── 8. RRC matched filter (I and Q) ──────────────────────────────────
		int Nr = (int)resampled_I.size();
		rrc_out_I.resize(Nr); rrc_out_Q.resize(Nr);
		rrc_combined_I.resize(rds_taps - 1 + Nr);
		rrc_combined_Q.resize(rds_taps - 1 + Nr);
		TIME_BLOCK("RDS_RRC", {
			blockConvolve_Decimate(rrc_out_I, resampled_I, rrc_coeff,
			                      rrc_state_I, rrc_combined_I, 1);
			blockConvolve_Decimate(rrc_out_Q, resampled_Q, rrc_coeff,
			                      rrc_state_Q, rrc_combined_Q, 1);
		});

		// ── 9. CDR warmup ────────────────────────────────────────────────────
		TIME_BLOCK("RDS_CDR", {
			if (cdr_phase < 0) {
				// Accumulate blocks for warmup
				warmup_I.insert(warmup_I.end(), rrc_out_I.begin(), rrc_out_I.end());
				warmup_Q.insert(warmup_Q.end(), rrc_out_Q.begin(), rrc_out_Q.end());
				warmup_count++;

				if (warmup_count >= WARMUP_BLOCKS) {
					// Search over one symbol period for best sampling phase
					real best_score = -1.0;
					int  best_phase = 0;
					bool best_useI  = true;
					for (int ph = 0; ph < SPS * 2; ph++) {
						real scoreI = 0.0, scoreQ = 0.0;
						int  cnt    = 0;
						for (int k = ph; k < (int)warmup_I.size(); k += SPS * 2) {
							scoreI += std::abs(warmup_I[k]);
							scoreQ += std::abs(warmup_Q[k]);
							cnt++;
						}
						if (cnt > 0) { scoreI /= cnt; scoreQ /= cnt; }
						if (scoreI > best_score) { best_score = scoreI; best_phase = ph; best_useI = true; }
						if (scoreQ > best_score) { best_score = scoreQ; best_phase = ph; best_useI = false; }
					}
					cdr_phase  = best_phase;
					use_I_axis = best_useI;
					warmup_I.clear(); warmup_Q.clear();
				}
			} else {
				// Steady-state: sample at cdr_phase within each symbol period
				const std::vector<real>& axis = use_I_axis ? rrc_out_I : rrc_out_Q;
				std::vector<int> symbols;
				for (int k = cdr_phase % (SPS * 2); k < Nr; k += SPS * 2) {
					symbols.push_back(axis[k] >= 0.0f ? 1 : -1);
				}
				cdr_phase = (cdr_phase + Nr) % (SPS * 2);

				// Manchester + differential decode
				auto bits_m = manchesterDecode(symbols);
				auto bits_d = differentialDecode(bits_m);
				bit_accumulator.insert(bit_accumulator.end(), bits_d.begin(), bits_d.end());

				// Process RDS data frame when we have enough bits
				if ((int)bit_accumulator.size() >= 104) {  // 4 blocks × 26 bits
					processRDSBits(bit_accumulator);
					// Keep a sliding window
					if ((int)bit_accumulator.size() > 208)
						bit_accumulator.erase(bit_accumulator.begin(),
						                      bit_accumulator.begin() + 104);
				}
			}
		});
	}
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
	int  mode        = 0;
	char path_flag   = 'm';
	int  filter_taps = 101;

	if (argc < 2) {
		std::cerr << "Operating in default mode 0 - mono path - filtertaps=101\n";
	} else if (argc == 3 || argc == 4) {
		mode        = atoi(argv[1]);
		path_flag   = argv[2][0];
		if (argc == 4) filter_taps = atoi(argv[3]);
		if (mode > 3) { std::cerr << "Wrong mode " << mode << "\n"; exit(1); }
	} else {
		std::cerr << "Usage: " << argv[0] << " <mode 0-3> <path m|s|r> [taps]\n";
		exit(1);
	}

	std::cerr << "Mode=" << mode
	          << "  Path=" << path_flag
	          << "  Taps=" << filter_taps << "\n";
	std::cerr << "Timing: will sample " << N_TIMING_BLOCKS << " RF blocks then report.\n";

	// RDS is only valid in modes 0 and 2
	bool do_rds = (path_flag == 'r') && (mode == 0 || mode == 2);
	if (path_flag == 'r' && !do_rds) {
		std::cerr << "[warn] RDS not supported in mode " << mode
		          << " – running stereo instead.\n";
		path_flag = 's';
	}

	// ── Build queues ─────────────────────────────────────────────────────────
	BoundedQueue<IFBlock> audioQueue(4);
	BoundedQueue<IFBlock> rdsQueue(4);

	std::atomic<bool> stopFlag{false};

	// ── Launch threads ────────────────────────────────────────────────────────
	char effective_audio_path = (path_flag == 'r') ? 's' : path_flag;

	std::thread rf_t(rfThread,
	                 mode, filter_taps,
	                 &audioQueue,
	                 do_rds ? &rdsQueue : nullptr,
	                 std::ref(stopFlag));

	std::thread audio_t(audioThread,
	                    mode, effective_audio_path, filter_taps,
	                    &audioQueue,
	                    std::ref(stopFlag));

	std::thread rds_t;
	if (do_rds) {
		rds_t = std::thread(rdsThread,
		                    mode, filter_taps,
		                    &rdsQueue,
		                    std::ref(stopFlag));
	}

	// ── Wait for all threads ──────────────────────────────────────────────────
	rf_t.join();
	audio_t.join();
	if (rds_t.joinable()) rds_t.join();

	// ── Print timing report ───────────────────────────────────────────────────
	printTimingSummary(mode, filter_taps);

	return 0;
}

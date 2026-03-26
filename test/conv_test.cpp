/*
   Comp Eng 3DY4 (Computer Systems Integration Project)

   Department of Electrical and Computer Engineering
   McMaster University
   Ontario, Canada
*/

// This file shows how to write convolution unit tests, using Google C++ test framework.
// (it is based on https://github.com/google/googletest/blob/main/docs/index.md)

#include <limits.h>
#include "dy4.h"
#include "iofunc.h"
#include "filter.h"
#include "gtest/gtest.h"

namespace {

	class Convolution_Fixture: public ::testing::Test {

		public:

			const int N = 1024;	// signal size
			const int M = 101;	// kernel size
			const int lower_bound = -1;
			const int upper_bound = 1;
			const real EPSILON = 1e-4;

			std::vector<real> x, h, y_reference, y_test;

			Convolution_Fixture() {
				x.resize(N);
				h.resize(M);
				y_reference.resize(N + M - 1);
				y_test.resize(N + M - 1);
			}

			void SetUp() {
				generate_random_values(x, lower_bound, upper_bound);
				generate_random_values(h, lower_bound, upper_bound);
				convolveFIR_reference(y_reference, x, h);
			}

			void TearDown() {
			}

			~Convolution_Fixture() {
			}
	};

	TEST_F(Convolution_Fixture, convolveFIR_inefficient_NEAR) {

		convolveFIR_inefficient(y_test, x, h);

		ASSERT_EQ(y_reference.size(), y_test.size()) << "Output vector sizes for convolveFIR_reference and convolveFIR_inefficient are unequal";

		for (int i = 0; i < (int)y_reference.size(); i++) {
			EXPECT_NEAR(y_reference[i], y_test[i], EPSILON) << "Original/convolveFIR_inefficient vectors differ at index " << i;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Unit tests: blockConvolve_DecimateFast vs blockConvolve_DecimateSlow
	//
	// Both functions implement overlap-save block convolution with decimation.
	// The "slow" version computes the full convolution first then keeps every
	// D-th output; the "fast" version uses loop unrolling and skips non-kept
	// outputs directly.  They must produce bit-for-bit equivalent results.
	// ─────────────────────────────────────────────────────────────────────────
	class DecimateBlock_Fixture : public ::testing::Test {
	public:
		// Block size matching a realistic mono audio path (mode 0, D=5):
		// IF block = 9600 samples → audio block = 9600/5 = 1920 samples
		const int N = 9600;
		const int M = 101;
		const int D = 5;
		const real EPSILON = 1e-3f;

		std::vector<real> x, h;

		DecimateBlock_Fixture() { x.resize(N); h.resize(M); }

		void SetUp() override {
			generate_random_values(x, -1.0f, 1.0f);
			generate_random_values(h, -1.0f, 1.0f);
		}
	};

	// Single block: fast and slow must agree.
	TEST_F(DecimateBlock_Fixture, single_block_fast_matches_slow) {
		std::vector<real> state_fast(M - 1, 0.0f), state_slow(M - 1, 0.0f);
		std::vector<real> comb_fast(M - 1 + N, 0.0f), comb_slow(M - 1 + N, 0.0f);

		// DecimateFast pre-allocates to exact output size N/D
		std::vector<real> y_fast(N / D, 0.0f);
		// DecimateSlow needs full-size N pre-allocation; it resizes internally
		std::vector<real> y_slow(N, 0.0f);

		blockConvolve_DecimateFast(y_fast, x, h, state_fast, comb_fast, D);
		blockConvolve_DecimateSlow(y_slow, x, h, state_slow, comb_slow, D);

		ASSERT_EQ(y_fast.size(), y_slow.size())
			<< "Output sizes differ: fast=" << y_fast.size()
			<< " slow=" << y_slow.size();
		for (int i = 0; i < (int)y_fast.size(); i++)
			EXPECT_NEAR(y_fast[i], y_slow[i], EPSILON)
				<< "Mismatch at output index " << i;
	}

	// Three consecutive blocks: state must carry over correctly in both versions.
	TEST_F(DecimateBlock_Fixture, multi_block_state_propagation) {
		std::vector<real> state_fast(M - 1, 0.0f), state_slow(M - 1, 0.0f);
		std::vector<real> comb_fast(M - 1 + N, 0.0f), comb_slow(M - 1 + N, 0.0f);

		for (int blk = 0; blk < 3; blk++) {
			// Fresh random input per block (like real streaming data)
			generate_random_values(x, -1.0f, 1.0f);

			std::vector<real> y_fast(N / D, 0.0f);
			std::vector<real> y_slow(N, 0.0f);

			blockConvolve_DecimateFast(y_fast, x, h, state_fast, comb_fast, D);
			blockConvolve_DecimateSlow(y_slow, x, h, state_slow, comb_slow, D);

			ASSERT_EQ(y_fast.size(), y_slow.size())
				<< "Block " << blk << ": output sizes differ";
			for (int i = 0; i < (int)y_fast.size(); i++)
				EXPECT_NEAR(y_fast[i], y_slow[i], EPSILON)
					<< "Block " << blk << ", index " << i;
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// Unit tests: blockConvolve_ResampleFast vs blockConvolve_ResampleSlow
	//
	// Both implement rational resampling (upsample by U, then decimate by D).
	// The "slow" version explicitly inserts zeros and convolves at the high rate;
	// the "fast" version uses a polyphase structure operating at the input rate.
	// They must produce equivalent outputs within floating-point tolerance.
	// ─────────────────────────────────────────────────────────────────────────
	class ResampleBlock_Fixture : public ::testing::Test {
	public:
		// Reduced block/filter sizes so the slow (upsampled) buffer stays small
		const int N = 256;    // input block size
		const int taps = 11;  // base filter taps
		const int U = 5;      // upsample factor
		const int D = 7;      // decimate factor
		// Full filter length = taps * U (standard for polyphase resamplers)
		int M;
		const real EPSILON = 1e-3f;

		std::vector<real> x, h;

		ResampleBlock_Fixture() : M(taps * U) {
			x.resize(N);
			h.resize(M);
		}

		void SetUp() override {
			generate_random_values(x, -1.0f, 1.0f);
			// Scale by U to match the gain normalisation used in project.cpp
			generate_random_values(h, -1.0f / U, 1.0f / U);
		}
	};

	// Single block: fast and slow must agree.
	TEST_F(ResampleBlock_Fixture, single_block_fast_matches_slow) {
		// Fast: combined at INPUT rate (M-1 + N)
		std::vector<real> state_fast(M - 1, 0.0f);
		std::vector<real> comb_fast(M - 1 + N, 0.0f);
		std::vector<real> y_fast(N * U / D, 0.0f);

		// Slow: combined at UPSAMPLED rate (M-1 + N*U)
		std::vector<real> state_slow(M - 1, 0.0f);
		std::vector<real> comb_slow(M - 1 + N * U, 0.0f);
		std::vector<real> y_slow(N * U, 0.0f);  // slow resizes internally

		blockConvolve_ResampleFast(y_fast, x, h, state_fast, comb_fast, D, U);
		blockConvolve_ResampleSlow(y_slow, x, h, state_slow, comb_slow, D, U);

		ASSERT_EQ(y_fast.size(), y_slow.size())
			<< "Output sizes differ: fast=" << y_fast.size()
			<< " slow=" << y_slow.size();
		for (int i = 0; i < (int)y_fast.size(); i++)
			EXPECT_NEAR(y_fast[i], y_slow[i], EPSILON)
				<< "Mismatch at output index " << i;
	}

	// Three consecutive blocks: state must carry over correctly in both versions.
	TEST_F(ResampleBlock_Fixture, multi_block_state_propagation) {
		std::vector<real> state_fast(M - 1, 0.0f);
		std::vector<real> comb_fast(M - 1 + N, 0.0f);

		std::vector<real> state_slow(M - 1, 0.0f);
		std::vector<real> comb_slow(M - 1 + N * U, 0.0f);

		for (int blk = 0; blk < 3; blk++) {
			generate_random_values(x, -1.0f, 1.0f);

			std::vector<real> y_fast(N * U / D, 0.0f);
			std::vector<real> y_slow(N * U, 0.0f);

			blockConvolve_ResampleFast(y_fast, x, h, state_fast, comb_fast, D, U);
			blockConvolve_ResampleSlow(y_slow, x, h, state_slow, comb_slow, D, U);

			ASSERT_EQ(y_fast.size(), y_slow.size())
				<< "Block " << blk << ": output sizes differ";
			for (int i = 0; i < (int)y_fast.size(); i++)
				EXPECT_NEAR(y_fast[i], y_slow[i], EPSILON)
					<< "Block " << blk << ", index " << i;
		}
	}

} // end of namespace

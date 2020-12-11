use std::arch::x86_64::*;

pub struct BiQuadSSE2 {
    c_xp3: __m128,
    c_xp2: __m128,
    c_xp1: __m128,
    c_x0: __m128,
    c_xm1: __m128,
    c_xm2: __m128,
    c_ym1: __m128,
    c_ym2: __m128,

    xm1: __m128,
    xm2: __m128,
    ym1: __m128,
    ym2: __m128,
}

impl BiQuadSSE2 {
    pub fn new() -> Self {
        unsafe {
            let mut b = BiQuadSSE2 {
                c_xp3: _mm_setzero_ps(),
                c_xp2: _mm_setzero_ps(),
                c_xp1: _mm_setzero_ps(),
                c_x0: _mm_setzero_ps(),
                c_xm1: _mm_setzero_ps(),
                c_xm2: _mm_setzero_ps(),
                c_ym1: _mm_setzero_ps(),
                c_ym2: _mm_setzero_ps(),
                xm1: _mm_setzero_ps(),
                xm2: _mm_setzero_ps(),
                ym1: _mm_setzero_ps(),
                ym2: _mm_setzero_ps(),
            };
            b.update(44100.0, 1200.0);
            b
        }
    }

    pub fn update(&mut self, sample_rate: f32, cutoff: f32) {
        unsafe {
            let t = (std::f32::consts::PI * cutoff / sample_rate).tan();
            let alpha = (t - 1.0) / (t + 1.0);

            let a0 = alpha;
            let a1 = 1.0;
            let a2 = 0.0;
            let b1 = alpha;
            let b2 = 0.0;

            const COLUMNS: usize = 8;
            const ROWS: usize = 4;

            #[rustfmt::skip]
            let mut coeffs: [[f32; COLUMNS]; ROWS] = [
                //                x[n+3] x[n+2] x[n+1]   x[n] x[n-1] x[n-2]  |  y[n-1] y[n-2]
                /* y[n]   = */ [    0.0,   0.0,   0.0,    a0,    a1,    a2,       -b1,   -b2],
                /* y[n+1] = */ [    0.0,   0.0,    a0,    a1,    a2,   0.0,       -b2,   0.0], // - b1 * y[n]
                /* y[n+2] = */ [    0.0,    a0,    a1,    a2,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-1] - b2 * y[n]
                /* y[n+3] = */ [     a0,    a1,    a2,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-2] - b2 * y[n-1]
            ];

            for col in 0..COLUMNS {
                // Add -b1 * y[n] to y[n+1] to y[n+1]
                coeffs[1][col] += -b1 * coeffs[0][col];

                // Add -b1 * y[n+1] - b2 * y[n] to y[n+2]
                coeffs[2][col] += -b1 * coeffs[1][col] + -b2 * coeffs[0][col];

                // Add -b1 * y[n+2] - b2 * y[n+1] to y[n+3]
                coeffs[3][col] += -b1 * coeffs[2][col] + -b2 * coeffs[1][col];
            }

            self.c_xp3 = _mm_set_ps(coeffs[3][0], coeffs[2][0], coeffs[1][0], coeffs[0][0]);
            self.c_xp2 = _mm_set_ps(coeffs[3][1], coeffs[2][1], coeffs[1][1], coeffs[0][1]);
            self.c_xp1 = _mm_set_ps(coeffs[3][2], coeffs[2][2], coeffs[1][2], coeffs[0][2]);
            self.c_x0 = _mm_set_ps(coeffs[3][3], coeffs[2][3], coeffs[1][3], coeffs[0][3]);
            self.c_xm1 = _mm_set_ps(coeffs[3][4], coeffs[2][4], coeffs[1][4], coeffs[0][4]);
            self.c_xm2 = _mm_set_ps(coeffs[3][5], coeffs[2][5], coeffs[1][5], coeffs[0][5]);
            self.c_ym1 = _mm_set_ps(coeffs[3][6], coeffs[2][6], coeffs[1][6], coeffs[0][6]);
            self.c_ym2 = _mm_set_ps(coeffs[3][7], coeffs[2][7], coeffs[1][7], coeffs[0][7]);
        }
    }

    pub fn process(&mut self, input: __m128) -> __m128 {
        unsafe {
            let v_x0 = _mm_shuffle_ps(input, input, 0b00_00_00_00);
            let v_xp1 = _mm_shuffle_ps(input, input, 0b01_01_01_01);
            let v_xp2 = _mm_shuffle_ps(input, input, 0b10_10_10_10);
            let v_xp3 = _mm_shuffle_ps(input, input, 0b11_11_11_11);

            let mut y = _mm_setzero_ps();
            y = _mm_fmadd_ps(self.c_xp3, v_xp3, y);
            y = _mm_fmadd_ps(self.c_xp2, v_xp2, y);
            y = _mm_fmadd_ps(self.c_xp1, v_xp1, y);
            y = _mm_fmadd_ps(self.c_x0, v_x0, y);
            y = _mm_fmadd_ps(self.c_xm1, self.xm1, y);
            y = _mm_fmadd_ps(self.c_xm2, self.xm2, y);
            y = _mm_fmadd_ps(self.c_ym1, self.ym1, y);
            y = _mm_fmadd_ps(self.c_ym2, self.ym2, y);

            self.xm2 = v_xp2;
            self.xm1 = v_xp3;
            self.ym2 = _mm_shuffle_ps(y, y, 0b10_10_10_10);
            self.ym1 = _mm_shuffle_ps(y, y, 0b11_11_11_11);

            y
        }
    }
}

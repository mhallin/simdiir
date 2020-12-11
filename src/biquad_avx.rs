use std::arch::x86_64::*;

pub struct BiQuadAVX {
    c_xp7: __m256,
    c_xp6: __m256,
    c_xp5: __m256,
    c_xp4: __m256,
    c_xp3: __m256,
    c_xp2: __m256,
    c_xp1: __m256,
    c_x0: __m256,
    c_xm1: __m256,
    c_xm2: __m256,
    c_ym1: __m256,
    c_ym2: __m256,

    xm1: __m256,
    xm2: __m256,
    ym1: __m256,
    ym2: __m256,
}

impl BiQuadAVX {
    pub fn new() -> Self {
        unsafe {
            let mut b = BiQuadAVX {
                c_xp7: _mm256_setzero_ps(),
                c_xp6: _mm256_setzero_ps(),
                c_xp5: _mm256_setzero_ps(),
                c_xp4: _mm256_setzero_ps(),
                c_xp3: _mm256_setzero_ps(),
                c_xp2: _mm256_setzero_ps(),
                c_xp1: _mm256_setzero_ps(),
                c_x0: _mm256_setzero_ps(),
                c_xm1: _mm256_setzero_ps(),
                c_xm2: _mm256_setzero_ps(),
                c_ym1: _mm256_setzero_ps(),
                c_ym2: _mm256_setzero_ps(),
                xm1: _mm256_setzero_ps(),
                xm2: _mm256_setzero_ps(),
                ym1: _mm256_setzero_ps(),
                ym2: _mm256_setzero_ps(),
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

            const COLUMNS: usize = 12;
            const ROWS: usize = 8;

            #[rustfmt::skip]
            let mut coeffs: [[f32; COLUMNS]; ROWS] = [
                //                x[n+7] x[n+6] x[n+5] x[n+4] x[n+3] x[n+2] x[n+1]   x[n] x[n-1] x[n-2]  |  y[n-1] y[n-2]
                /* y[n]   = */ [    0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,    a0,    a1,    a2,       -b1,   -b2],
                /* y[n+1] = */ [    0.0,   0.0,   0.0,   0.0,   0.0,   0.0,    a0,    a1,    a2,   0.0,       -b2,   0.0], // - b1 * y[n]
                /* y[n+2] = */ [    0.0,   0.0,   0.0,   0.0,   0.0,    a0,    a1,    a2,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-1] - b2 * y[n]
                /* y[n+3] = */ [    0.0,   0.0,   0.0,   0.0,    a0,    a1,    a2,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-2] - b2 * y[n-1]
                /* y[n+4] = */ [    0.0,   0.0,   0.0,    a0,    a1,    a2,   0.0,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-3] - b2 * y[n-2]
                /* y[n+5] = */ [    0.0,   0.0,    a0,    a1,    a2,   0.0,   0.0,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-4] - b2 * y[n-3]
                /* y[n+6] = */ [    0.0,    a0,    a1,    a2,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-5] - b2 * y[n-4]
                /* y[n+7] = */ [     a0,    a1,    a2,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,       0.0,   0.0], // - b1 * y[n-6] - b2 * y[n-5]
            ];

            for col in 0..COLUMNS {
                // Add -b1 * y[n] to y[n+1] to y[n+1]
                coeffs[1][col] += -b1 * coeffs[0][col];

                // Add -b1 * y[n+1] - b2 * y[n] to y[n+2]
                coeffs[2][col] += -b1 * coeffs[1][col] + -b2 * coeffs[0][col];

                // Add -b1 * y[n+2] - b2 * y[n+1] to y[n+3]
                coeffs[3][col] += -b1 * coeffs[2][col] + -b2 * coeffs[1][col];

                // Add -b1 * y[n+3] - b2 * y[n+1] to y[n+4]
                coeffs[4][col] += -b1 * coeffs[3][col] + -b2 * coeffs[2][col];

                // Add -b1 * y[n+4] - b2 * y[n+1] to y[n+5]
                coeffs[5][col] += -b1 * coeffs[4][col] + -b2 * coeffs[3][col];

                // Add -b1 * y[n+5] - b2 * y[n+1] to y[n+6]
                coeffs[6][col] += -b1 * coeffs[5][col] + -b2 * coeffs[4][col];

                // Add -b1 * y[n+6] - b2 * y[n+1] to y[n+7]
                coeffs[7][col] += -b1 * coeffs[6][col] + -b2 * coeffs[5][col];
            }

            self.c_xp7 = _mm256_set_ps(
                coeffs[7][0],
                coeffs[6][0],
                coeffs[5][0],
                coeffs[4][0],
                coeffs[3][0],
                coeffs[2][0],
                coeffs[1][0],
                coeffs[0][0],
            );
            self.c_xp6 = _mm256_set_ps(
                coeffs[7][1],
                coeffs[6][1],
                coeffs[5][1],
                coeffs[4][1],
                coeffs[3][1],
                coeffs[2][1],
                coeffs[1][1],
                coeffs[0][1],
            );
            self.c_xp5 = _mm256_set_ps(
                coeffs[7][2],
                coeffs[6][2],
                coeffs[5][2],
                coeffs[4][2],
                coeffs[3][2],
                coeffs[2][2],
                coeffs[1][2],
                coeffs[0][2],
            );
            self.c_xp4 = _mm256_set_ps(
                coeffs[7][3],
                coeffs[6][3],
                coeffs[5][3],
                coeffs[4][3],
                coeffs[3][3],
                coeffs[2][3],
                coeffs[1][3],
                coeffs[0][3],
            );
            self.c_xp3 = _mm256_set_ps(
                coeffs[7][4],
                coeffs[6][4],
                coeffs[5][4],
                coeffs[4][4],
                coeffs[3][4],
                coeffs[2][4],
                coeffs[1][4],
                coeffs[0][4],
            );
            self.c_xp2 = _mm256_set_ps(
                coeffs[7][5],
                coeffs[6][5],
                coeffs[5][5],
                coeffs[4][5],
                coeffs[3][5],
                coeffs[2][5],
                coeffs[1][5],
                coeffs[0][5],
            );
            self.c_xp1 = _mm256_set_ps(
                coeffs[7][6],
                coeffs[6][6],
                coeffs[5][6],
                coeffs[4][6],
                coeffs[3][6],
                coeffs[2][6],
                coeffs[1][6],
                coeffs[0][6],
            );
            self.c_x0 = _mm256_set_ps(
                coeffs[7][7],
                coeffs[6][7],
                coeffs[5][7],
                coeffs[4][7],
                coeffs[3][7],
                coeffs[2][7],
                coeffs[1][7],
                coeffs[0][7],
            );
            self.c_xm1 = _mm256_set_ps(
                coeffs[7][8],
                coeffs[6][8],
                coeffs[5][8],
                coeffs[4][8],
                coeffs[3][8],
                coeffs[2][8],
                coeffs[1][8],
                coeffs[0][8],
            );
            self.c_xm2 = _mm256_set_ps(
                coeffs[7][9],
                coeffs[6][9],
                coeffs[5][9],
                coeffs[4][9],
                coeffs[3][9],
                coeffs[2][9],
                coeffs[1][9],
                coeffs[0][9],
            );
            self.c_ym1 = _mm256_set_ps(
                coeffs[7][10],
                coeffs[6][10],
                coeffs[5][10],
                coeffs[4][10],
                coeffs[3][10],
                coeffs[2][10],
                coeffs[1][10],
                coeffs[0][10],
            );
            self.c_ym2 = _mm256_set_ps(
                coeffs[7][11],
                coeffs[6][11],
                coeffs[5][11],
                coeffs[4][11],
                coeffs[3][11],
                coeffs[2][11],
                coeffs[1][11],
                coeffs[0][11],
            );
        }
    }

    pub fn process(&mut self, input: __m256) -> __m256 {
        unsafe {
            let v_x0 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(0));
            let v_xp1 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(1));
            let v_xp2 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(2));
            let v_xp3 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(3));
            let v_xp4 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(4));
            let v_xp5 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(5));
            let v_xp6 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(6));
            let v_xp7 = _mm256_permutevar8x32_ps(input, _mm256_set1_epi32(7));

            let mut y1 = _mm256_setzero_ps();
            let mut y2 = _mm256_setzero_ps();
            y1 = _mm256_fmadd_ps(self.c_xp7, v_xp7, y1);
            y2 = _mm256_fmadd_ps(self.c_xp6, v_xp6, y2);
            y1 = _mm256_fmadd_ps(self.c_xp5, v_xp5, y1);
            y2 = _mm256_fmadd_ps(self.c_xp4, v_xp4, y2);
            y1 = _mm256_fmadd_ps(self.c_xp3, v_xp3, y1);
            y2 = _mm256_fmadd_ps(self.c_xp2, v_xp2, y2);
            y1 = _mm256_fmadd_ps(self.c_xp1, v_xp1, y1);
            y2 = _mm256_fmadd_ps(self.c_x0, v_x0, y2);
            y1 = _mm256_fmadd_ps(self.c_xm1, self.xm1, y1);
            y2 = _mm256_fmadd_ps(self.c_xm2, self.xm2, y2);
            y1 = _mm256_fmadd_ps(self.c_ym1, self.ym1, y1);
            y2 = _mm256_fmadd_ps(self.c_ym2, self.ym2, y2);

            let y = _mm256_add_ps(y1, y2);

            self.xm2 = v_xp2;
            self.xm1 = v_xp3;
            self.ym2 = _mm256_shuffle_ps(y, y, 0b10_10_10_10);
            self.ym1 = _mm256_shuffle_ps(y, y, 0b11_11_11_11);

            y
        }
    }
}

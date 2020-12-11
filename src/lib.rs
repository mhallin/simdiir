pub mod biquad_f32;
pub mod biquad_sse2;
pub mod biquad_avx;

pub struct ScopedFlushDenormals {
    _hidden: (),
}

impl ScopedFlushDenormals {
    pub fn new() -> Self {
        use std::arch::x86_64;
        unsafe {
            x86_64::_MM_SET_FLUSH_ZERO_MODE(x86_64::_MM_FLUSH_ZERO_ON);
        }

        ScopedFlushDenormals { _hidden: () }
    }
}

impl Drop for ScopedFlushDenormals {
    fn drop(&mut self) {
        use std::arch::x86_64;
        unsafe {
            x86_64::_MM_SET_FLUSH_ZERO_MODE(x86_64::_MM_FLUSH_ZERO_OFF);
        }
    }
}

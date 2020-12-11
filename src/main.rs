use simdiir::{biquad_avx::BiQuadAVX, biquad_f32::BiQuadF32, biquad_sse2::BiQuadSSE2};

fn main() {
    let impulse = {
        let mut data = vec![0.0; 256];
        data[0] = 1.0;
        data
    };

    let bf32 = {
        let mut b = BiQuadF32::new();
        let mut output = vec![0.0; impulse.len()];
        run_f32(&mut b, &impulse, &mut output);
        output
    };

    let bf32_no_denorm = {
        let mut b = BiQuadF32::new();
        let mut output = vec![0.0; impulse.len()];
        run_f32_no_denorm(&mut b, &impulse, &mut output);
        output
    };

    let sse2 = {
        let mut b = BiQuadSSE2::new();
        let mut output = vec![0.0; impulse.len()];
        run_sse2(&mut b, &impulse, &mut output);
        output
    };

    let avx = {
        let mut b = BiQuadAVX::new();
        let mut output = vec![0.0; impulse.len()];
        run_avx(&mut b, &impulse, &mut output);
        output
    };

    println!("input,f32 denorm,f32 flush denorm,sse2,avx");
    for ((((input, o_f32), o_f32_f), o_sse2), o_avx) in impulse
        .iter()
        .zip(bf32.iter())
        .zip(bf32_no_denorm.iter())
        .zip(sse2.iter())
        .zip(avx.iter())
    {
        println!("{},{},{},{},{}", input, o_f32, o_f32_f, o_sse2, o_avx);
    }
}

fn run_f32(b: &mut BiQuadF32, input: &[f32], output: &mut [f32]) {
    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = b.process(*input)
    }
}

fn run_f32_no_denorm(b: &mut BiQuadF32, input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64;
    unsafe {
        x86_64::_MM_SET_FLUSH_ZERO_MODE(x86_64::_MM_FLUSH_ZERO_ON);
    }

    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = b.process(*input)
    }

    unsafe {
        x86_64::_MM_SET_FLUSH_ZERO_MODE(x86_64::_MM_FLUSH_ZERO_OFF);
    }
}

fn run_sse2(b: &mut BiQuadSSE2, input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::__m128;
    let input = unsafe {
        let ptr = input.as_ptr() as *const __m128;
        std::slice::from_raw_parts(ptr, input.len() / 4)
    };
    let output = unsafe {
        let ptr = output.as_mut_ptr() as *mut __m128;
        std::slice::from_raw_parts_mut(ptr, output.len() / 4)
    };

    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = b.process(*input);
    }
}

fn run_avx(b: &mut BiQuadAVX, input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::__m256;
    let input = unsafe {
        let ptr = input.as_ptr() as *const __m256;
        std::slice::from_raw_parts(ptr, input.len() / 8)
    };
    let output = unsafe {
        let ptr = output.as_mut_ptr() as *mut __m256;
        std::slice::from_raw_parts_mut(ptr, output.len() / 8)
    };

    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = b.process(*input);
    }
}

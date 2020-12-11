use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup, Criterion,
};
use rand::{Rng, SeedableRng};

use simdiir::{
    biquad_avx::BiQuadAVX, biquad_f32::BiQuadF32, biquad_sse2::BiQuadSSE2, ScopedFlushDenormals,
};

fn noise(c: &mut Criterion) {
    let mut rng = rand_xorshift::XorShiftRng::seed_from_u64(3_031_657_322_766_356_513);
    let noise_data = (0..65536)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("Noise input");
    run_tests_with_input(&noise_data, &mut group);
}

fn impulse(c: &mut Criterion) {
    let mut impulse = vec![0.0; 65536];
    impulse[0] = 1.0;

    let mut group = c.benchmark_group("Impulse");
    run_tests_with_input(&impulse, &mut group);
}

criterion_group!(benches, noise, impulse);
criterion_main!(benches);

fn run_tests_with_input(input: &[f32], group: &mut BenchmarkGroup<WallTime>) {
    group.bench_with_input("f32 with denorm", &input, |b, input| {
        let mut biquad = BiQuadF32::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| run_f32_bench(&mut biquad, &input, &mut output),
            BatchSize::PerIteration,
        );
    });

    group.bench_with_input("f32 flush denorm", &input, |b, input| {
        let mut biquad = BiQuadF32::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| {
                let _guard = ScopedFlushDenormals::new();
                run_f32_bench(&mut biquad, &input, &mut output)
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_with_input("sse2 with denorm", &input, |b, input| {
        let mut biquad = BiQuadSSE2::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| run_sse2_bench(&mut biquad, &input, &mut output),
            BatchSize::PerIteration,
        );
    });

    group.bench_with_input("sse2 flush denorm", &input, |b, input| {
        let mut biquad = BiQuadSSE2::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| {
                let _guard = ScopedFlushDenormals::new();
                run_sse2_bench(&mut biquad, &input, &mut output)
            },
            BatchSize::PerIteration,
        );
    });

    group.bench_with_input("avx2 with denorm", &input, |b, input| {
        let mut biquad = BiQuadAVX::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| run_avx_bench(&mut biquad, &input, &mut output),
            BatchSize::PerIteration,
        );
    });

    group.bench_with_input("avx2 flush denorm", &input, |b, input| {
        let mut biquad = BiQuadAVX::new();
        b.iter_batched_ref(
            || vec![0.0; input.len()],
            |mut output| {
                let _guard = ScopedFlushDenormals::new();
                run_avx_bench(&mut biquad, &input, &mut output)
            },
            BatchSize::PerIteration,
        );
    });
}

fn run_f32_bench(biquad: &mut BiQuadF32, input: &[f32], output: &mut [f32]) {
    for (input, output) in input.iter().zip(output.iter_mut()) {
        *output = biquad.process(*input);
    }
}

fn run_sse2_bench(biquad: &mut BiQuadSSE2, input: &[f32], output: &mut [f32]) {
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
        *output = biquad.process(*input);
    }
}

fn run_avx_bench(biquad: &mut BiQuadAVX, input: &[f32], output: &mut [f32]) {
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
        *output = biquad.process(*input);
    }
}

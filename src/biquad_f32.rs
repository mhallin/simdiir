#[derive(Copy, Clone)]
pub struct BiQuadF32 {
    pub a0: f32,
    pub a1: f32,
    pub a2: f32,
    pub b1: f32,
    pub b2: f32,

    z_a1: f32,
    z_a2: f32,
    z_b1: f32,
    z_b2: f32,
}

impl BiQuadF32 {
    pub fn new() -> Self {
        let mut b = Self {
            a0: 0.0,
            a1: 0.0,
            a2: 0.0,
            b1: 0.0,
            b2: 0.0,
            z_a1: 0.0,
            z_a2: 0.0,
            z_b1: 0.0,
            z_b2: 0.0,
        };
        b.update(44100.0, 1200.0);
        b
    }

    pub fn update(&mut self, sample_rate: f32, cutoff: f32) {
        let t = (std::f32::consts::PI * cutoff / sample_rate).tan();
        let alpha = (t - 1.0) / (t + 1.0);

        self.a0 = alpha;
        self.a1 = 1.0;
        self.a2 = 0.0;
        self.b1 = alpha;
        self.b2 = 0.0;
    }

    pub fn process(&mut self, input: f32) -> f32 {
        let xn = input;

        let yn = self.a0 * xn + self.a1 * self.z_a1 + self.a2 * self.z_a2
            - self.b1 * self.z_b1
            - self.b2 * self.z_b2;

        self.z_b2 = self.z_b1;
        self.z_b1 = yn;

        self.z_a2 = self.z_a1;
        self.z_a1 = xn;

        yn
    }
}

use ndarray::{Array1, Array2};
use crate::visitor::ParamVisitor;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct XavierInitializer {
    rng: rand::rngs::ThreadRng,
}

impl XavierInitializer {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl ParamVisitor for XavierInitializer {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<f64>,
        _grad: &Array2<f64>,
    ) {
        let (rows, cols) = param.dim();
        let std_dev = (2.0 / (rows + cols) as f64).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        for val in param.iter_mut() {
            *val = normal.sample(&mut self.rng);
        }
    }

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<f64>,
        _grad: &Array1<f64>,
    ) {
        param.fill(0.0);
    }
}

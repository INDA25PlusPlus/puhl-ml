use nalgebra::{SMatrix};

pub trait LossFunction<const B: usize, const N: usize> {
    fn forward(&mut self, prediction: &SMatrix<f64, N, B>, target: &SMatrix<f64, N, B>) -> f64;
    fn backward(&self) -> SMatrix<f64, N, B>;
}

pub struct MSE<const B: usize, const N: usize> {
    last_prediction: Option<SMatrix<f64, N, B>>,
    last_target: Option<SMatrix<f64, N, B>>,
}

impl<const B: usize, const N: usize> MSE<B, N> {
    pub fn new() -> Self {
        Self {
            last_prediction: None,
            last_target: None,
        }
    }
}

impl<const B: usize, const N: usize> LossFunction<B, N> for MSE<B, N> {
    fn forward(&mut self, prediction: &SMatrix<f64, N, B>, target: &SMatrix<f64, N, B>) -> f64 {
        self.last_prediction = Some(*prediction);
        self.last_target = Some(*target);

        let diff = prediction - target;

        diff.norm_squared() / (N * B) as f64
    }

    fn backward(&self) -> SMatrix<f64, N, B> {
        let prediction = self.last_prediction.as_ref()
            .expect("forward() must be called before backward()");
        let target = self.last_target.as_ref()
            .expect("forward() must be called before backward()");

        (2.0 / (N * B) as f64) * (prediction - target)
    }
}
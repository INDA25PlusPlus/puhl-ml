use nalgebra::SVector;

pub trait LossFunction<const N: usize> {
    fn forward(&mut self, prediction: &SVector<f64, N>, target: &SVector<f64, N>) -> f64;
    fn backward(&self) -> SVector<f64, N>;
}

pub struct MSE<const N: usize> {
    last_prediction: Option<SVector<f64, N>>,
    last_target: Option<SVector<f64, N>>,
}

impl<const N: usize> MSE<N> {
    pub fn new() -> Self {
        Self {
            last_prediction: None,
            last_target: None,
        }
    }
}

impl<const N: usize> LossFunction<N> for MSE<N> {
    fn forward(&mut self, prediction: &SVector<f64, N>, target: &SVector<f64, N>) -> f64 {
        self.last_prediction = Some(*prediction);
        self.last_target = Some(*target);

        let diff = prediction - target;

        diff.norm_squared() / N as f64
    }

    fn backward(&self) -> SVector<f64, N> {
        let prediction = self.last_prediction.as_ref()
            .expect("forward() must be called before backward()");
        let target = self.last_target.as_ref()
            .expect("forward() must be called before backward()");

        (2.0 / N as f64) * (prediction - target)
    }
}
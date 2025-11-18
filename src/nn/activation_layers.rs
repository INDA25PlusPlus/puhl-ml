use nalgebra::{SMatrix};
use super::layer::Layer;

pub struct ReLU<const B: usize, const I: usize> {
    last_input: Option<SMatrix<f64, I, B>>,
}

impl<const B: usize, const I: usize> ReLU<B, I> {
    pub fn new() -> Self {
        Self { last_input: None }
    }
}

impl<const B: usize, const I: usize> Layer for ReLU<B, I> {
    type Input = SMatrix<f64, I, B>;
    type Output = SMatrix<f64, I, B>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        self.last_input = Some(*input);
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let input = self.last_input.as_ref()
            .expect("forward() must be called before backward()");

        grad_output.zip_map(input, |grad, x| {
            if x > 0.0 { grad } else { 0.0 }
        })
    }
}
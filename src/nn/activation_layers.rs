use nalgebra::{SMatrix, SVector};

use super::layer::Layer;

pub struct ReLU<const I: usize> {
    last_input: Option<SVector<f64, I>>,
}

impl<const I: usize> Layer for ReLU<I> {
    type Input = SVector<f64, I>;
    type Output = SVector<f64, I>;

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
use ndarray::Array2;
use crate::layer::Layer;

pub struct ReLU {
    last_input: Option<Array2<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { last_input: None }
    }
}

impl Layer for ReLU {
    type Input = Array2<f64>;
    type Output = Array2<f64>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        self.last_input = Some(input.clone());
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let input = self.last_input.as_ref()
            .expect("forward() must be called before backward()");

        grad_output * &input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
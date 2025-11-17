use nalgebra::{SMatrix, SVector};
use super::visitor::ParamVisitor;

pub trait Layer {
    type Input;
    type Output;

    fn forward(&mut self, input: &Self::Input) -> Self::Output;
    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input;
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V);
}

pub struct LinearLayer<const I: usize, const O: usize> {
    pub weights: SMatrix<f64, O, I>,
    pub bias: SVector<f64, O>,

    pub weight_grad: SMatrix<f64, O, I>,
    pub bias_grad: SVector<f64, O>,

    last_input: Option<SVector<f64, I>>,
}

impl<const I: usize, const O: usize> LinearLayer<I, O> {
    pub fn new(weights: SMatrix<f64, O, I>, bias: SVector<f64, O>) -> Self {
        Self {
            weights,
            bias,
            weight_grad: SMatrix::zeros(),
            bias_grad: SVector::zeros(),
            last_input: None,
        }
    }

    pub fn zero_grad(&mut self) {
        self.weight_grad.fill(0.0);
        self.bias_grad.fill(0.0);
    }
}

impl<const I: usize, const O: usize> Layer for LinearLayer<I, O> {
    type Input = SVector<f64, I>;
    type Output = SVector<f64, O>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        self.last_input = Some(*input);
        &self.weights * input + &self.bias
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let input = self.last_input.as_ref()
            .expect("forward() must be called before backward()");

        self.weight_grad += grad_output * input.transpose();
        self.bias_grad += grad_output;

        self.weights.transpose() * grad_output
    }

    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_matrix(&mut self.weights);
        visitor.visit_vector(&mut self.bias);
    }
}
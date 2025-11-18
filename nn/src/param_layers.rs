use nalgebra::{SMatrix, SVector};
use crate::layer::Layer;
use crate::visitor::{ParamVisitor, Parameterized};

pub struct LinearLayer<const B: usize, const I: usize, const O: usize> {
    pub weights: SMatrix<f64, O, I>,
    pub bias: SVector<f64, O>,

    pub weight_grad: SMatrix<f64, O, I>,
    pub bias_grad: SVector<f64, O>,

    last_input: Option<SMatrix<f64, I, B>>,
}

impl<const B: usize, const I: usize, const O: usize> LinearLayer<B, I, O> {
    pub fn new(weights: SMatrix<f64, O, I>, bias: SVector<f64, O>) -> Self {
        Self {
            weights,
            bias,
            weight_grad: SMatrix::zeros(),
            bias_grad: SVector::zeros(),
            last_input: None,
        }
    }
}

impl<const B: usize, const I: usize, const O: usize> Layer for LinearLayer<B, I, O> {
    type Input = SMatrix<f64, I, B>;
    type Output = SMatrix<f64, O, B>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        self.last_input = Some(*input);
        &self.weights * input + (&self.bias * SVector::<f64, B>::repeat(1.0).transpose())
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let input = self.last_input.as_ref()
            .expect("forward() must be called before backward()");

        self.weight_grad += (grad_output * input.transpose()) / B as f64;
        self.bias_grad += (grad_output * SVector::<f64, B>::repeat(1.0)) / B as f64;

        self.weights.transpose() * grad_output
    }
}

impl<const B: usize, const I: usize, const O: usize> Parameterized for LinearLayer<B, I, O> {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_matrix_with_grad(&mut self.weights, &self.weight_grad);
        visitor.visit_vector_with_grad(&mut self.bias, &self.bias_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grad.fill(0.0);
        self.bias_grad.fill(0.0);
    }
}

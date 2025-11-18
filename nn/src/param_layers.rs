use ndarray::{Array1, Array2, Axis};
use crate::layer::Layer;
use crate::visitor::{ParamVisitor, Parameterized};

pub struct LinearLayer {
    pub weights: Array2<f64>,  // Shape: (output_features, input_features)
    pub bias: Array1<f64>,      // Shape: (output_features,)

    pub weight_grad: Array2<f64>,
    pub bias_grad: Array1<f64>,

    last_input: Option<Array2<f64>>,
}

impl LinearLayer {
    pub fn new(output_features: usize, input_features: usize) -> Self {
        Self {
            weights: Array2::zeros((output_features, input_features)),
            bias: Array1::zeros(output_features),
            weight_grad: Array2::zeros((output_features, input_features)),
            bias_grad: Array1::zeros(output_features),
            last_input: None,
        }
    }
}

impl Layer for LinearLayer {
    type Input = Array2<f64>;   // Shape: (input_features, batch_size)
    type Output = Array2<f64>;  // Shape: (output_features, batch_size)

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        self.last_input = Some(input.clone());

        let output = self.weights.dot(input);
        &output + &self.bias.view().insert_axis(Axis(1))
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let input = self.last_input.as_ref()
            .expect("forward() must be called before backward()");

        let batch_size = input.shape()[1] as f64;

        self.weight_grad = grad_output.dot(&input.t()) / batch_size;
        self.bias_grad = grad_output.sum_axis(Axis(1)) / batch_size;
        self.weights.t().dot(grad_output)
    }
}

impl Parameterized for LinearLayer {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_array2_with_grad(&mut self.weights, &self.weight_grad);
        visitor.visit_array1_with_grad(&mut self.bias, &self.bias_grad);
    }

    fn zero_grad(&mut self) {
        self.weight_grad.fill(0.0);
        self.bias_grad.fill(0.0);
    }
}

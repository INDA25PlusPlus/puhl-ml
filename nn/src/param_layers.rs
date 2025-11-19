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

        self.weight_grad = grad_output.dot(&input.t());
        self.bias_grad = grad_output.sum_axis(Axis(1));
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

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 2;
    const O: usize = 3;

    struct ParamInitializer {
        weight_lst: Array2<f64>,
        bias_lst: Array1<f64>,
    }

    impl ParamVisitor for ParamInitializer {
        fn visit_array1_with_grad(
                &mut self,
                param: &mut Array1<f64>,
                _grad: &Array1<f64>,
            ) {
            param.assign(&self.bias_lst);
        }

        fn visit_array2_with_grad(
                &mut self,
                param: &mut Array2<f64>,
                _grad: &Array2<f64>,
            ) {
            param.assign(&self.weight_lst);
        }
    }

    #[test]
    fn test_param_visitor_initialization() {
        let mut layer = LinearLayer::new(O, I);

        let custom_weights = Array2::from_shape_vec((O, I), vec![
            1.0, 2.0, 
            3.0, 4.0,    
            5.0, 6.0,    
        ]).unwrap();
        let custom_bias = Array1::from_vec(vec![10.0, 20.0, 30.0]);

        // Initialize with visitor
        let mut initializer = ParamInitializer {
            weight_lst: custom_weights.clone(),
            bias_lst: custom_bias.clone(),
        };
        layer.visit_params(&mut initializer);

        // Verify parameters were set
        assert_eq!(layer.weights, custom_weights);
        assert_eq!(layer.bias, custom_bias);

        // Test forward pass with 2 batches
        let input = Array2::from_shape_vec((I, 2), vec![
            2.0, 1.0,  
            3.0, 4.0,
        ]).unwrap();
        let output = layer.forward(&input);

        assert_eq!(output.shape(), &[O, 2]);
        assert!((output[[0, 0]] - 18.0).abs() < 1e-10);
        assert!((output[[1, 0]] - 38.0).abs() < 1e-10);
        assert!((output[[2, 0]] - 58.0).abs() < 1e-10);
        assert!((output[[0, 1]] - 19.0).abs() < 1e-10);
        assert!((output[[1, 1]] - 39.0).abs() < 1e-10);
        assert!((output[[2, 1]] - 59.0).abs() < 1e-10);

        // Test backward pass
        let grad_output = Array2::from_shape_vec((O, 2), vec![
            1.0, 2.0,  
            1.0, 2.0,  
            1.0, 2.0,  
        ]).unwrap();
        let grad_input = layer.backward(&grad_output);

        assert_eq!(grad_input.shape(), &[I, 2]);
        assert!((grad_input[[0, 0]] - 9.0).abs() < 1e-10);
        assert!((grad_input[[1, 0]] - 12.0).abs() < 1e-10);
        assert!((grad_input[[0, 1]] - 18.0).abs() < 1e-10);
        assert!((grad_input[[1, 1]] - 24.0).abs() < 1e-10);

        // Check weight gradients
        assert!((layer.weight_grad[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((layer.weight_grad[[0, 1]] - 5.5).abs() < 1e-10);
        assert!((layer.weight_grad[[1, 0]] - 2.0).abs() < 1e-10);
        assert!((layer.weight_grad[[1, 1]] - 5.5).abs() < 1e-10);
        assert!((layer.weight_grad[[2, 0]] - 2.0).abs() < 1e-10);
        assert!((layer.weight_grad[[2, 1]] - 5.5).abs() < 1e-10);

        // Check bias gradients 
        assert!((layer.bias_grad[0] - 1.5).abs() < 1e-10);
        assert!((layer.bias_grad[1] - 1.5).abs() < 1e-10);
        assert!((layer.bias_grad[2] - 1.5).abs() < 1e-10);
    }
}
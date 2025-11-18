use nalgebra::SVector;
use crate::nn::{activation_layers::ReLU, layer::Layer, param_layers::LinearLayer, visitor::Parameterized};

pub struct Network {
    layer1: LinearLayer<3, 4>,
    relu: ReLU<4>,
    layer2: LinearLayer<4, 1>,
}

impl Layer for Network {
    type Input = SVector<f64, 3>;
    type Output = SVector<f64, 1>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        let x = self.layer1.forward(input);
        let x = self.relu.forward(&x);
        self.layer2.forward(&x)
    } 

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let grad = self.layer2.backward(grad_output);
        let grad = self.relu.backward(&grad);
        self.layer1.backward(&grad)
    }
}

impl Parameterized for Network {
    fn visit_params<V: super::visitor::ParamVisitor>(&mut self, visitor: &mut V) {
        self.layer1.visit_params(visitor); 
        self.layer2.visit_params(visitor);
    }

    fn zero_grad(&mut self) {
        self.layer1.zero_grad();
        self.layer2.zero_grad();
    }
}
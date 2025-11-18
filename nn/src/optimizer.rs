use ndarray::{Array1, Array2};
use crate::visitor::ParamVisitor;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl ParamVisitor for SGD {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<f64>,
        grad: &Array2<f64>,
    ) {
        *param = &*param - &(grad * self.learning_rate);
    }

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<f64>,
        grad: &Array1<f64>,
    ) {
        *param = &*param - &(grad * self.learning_rate);
    }
}
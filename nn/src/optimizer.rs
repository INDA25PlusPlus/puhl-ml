use ndarray::{Array1, Array2};
use crate::visitor::ParamVisitor;
use crate::Float;

pub struct SGD {
    pub learning_rate: Float,
}

impl SGD {
    pub fn new(learning_rate: Float) -> Self {
        Self { learning_rate }
    }
}

impl ParamVisitor for SGD {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<Float>,
        grad: &Array2<Float>,
    ) {
        *param = &*param - &(grad * self.learning_rate);
    }

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<Float>,
        grad: &Array1<Float>,
    ) {
        *param = &*param - &(grad * self.learning_rate);
    }
}
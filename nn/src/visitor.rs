use ndarray::{Array1, Array2};
use crate::Float;

pub trait ParamVisitor {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<Float>,
        grad: &Array2<Float>,
    );

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<Float>,
        grad: &Array1<Float>,
    );
}

pub trait Parameterized {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V);
    fn zero_grad(&mut self);
}
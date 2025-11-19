use ndarray::{Array1, Array2};

pub trait ParamVisitor {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<f64>,
        grad: &Array2<f64>,
    );

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<f64>,
        grad: &Array1<f64>,
    );
}

pub trait Parameterized {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V);
    fn zero_grad(&mut self);
}
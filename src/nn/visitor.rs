use nalgebra::{SMatrix, SVector};

pub trait ParamVisitor {
    fn visit_matrix<const R: usize, const C: usize>(&mut self, param: &mut SMatrix<f64, R, C>);
    fn visit_vector<const N: usize>(&mut self, param: &mut SVector<f64, N>);
}

pub trait Parameterized {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V);
    fn zero_grad(&mut self);
}

use nalgebra::{SMatrix, SVector};

pub trait ParamVisitor {
    fn visit_matrix_with_grad<const R: usize, const C: usize>(
        &mut self,
        param: &mut SMatrix<f64, R, C>,
        grad: &SMatrix<f64, R, C>,
    );

    fn visit_vector_with_grad<const N: usize>(
        &mut self,
        param: &mut SVector<f64, N>,
        grad: &SVector<f64, N>,
    );
}

pub trait Parameterized {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V);
    fn zero_grad(&mut self);
}

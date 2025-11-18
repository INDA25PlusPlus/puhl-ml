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
    fn visit_matrix_with_grad<const R: usize, const C: usize>(
            &mut self,
            param: &mut nalgebra::SMatrix<f64, R, C>,
            grad: &nalgebra::SMatrix<f64, R, C>,
        ) {
        *param -= grad * self.learning_rate;
    }

    fn visit_vector_with_grad<const N: usize>(
            &mut self,
            param: &mut nalgebra::SVector<f64, N>,
            grad: &nalgebra::SVector<f64, N>,
        ) {
        *param -= grad * self.learning_rate;
    }
}
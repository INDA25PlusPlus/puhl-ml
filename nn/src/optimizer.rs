use ndarray::{azip, Array1, Array2};
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

pub struct SGDMomentum {
    pub learning_rate: Float,
    pub momentum: Float,      // Usually 0.9
    
    velocities_2d: Vec<Array2<Float>>,
    velocities_1d: Vec<Array1<Float>>,
    
    idx_2d: usize,
    idx_1d: usize,
}

impl SGDMomentum {
    pub fn new(learning_rate: Float, momentum: Float) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities_2d: Vec::new(),
            velocities_1d: Vec::new(),
            idx_2d: 0,
            idx_1d: 0,
        }
    }

    pub fn start_pass(&mut self) {
        self.idx_2d = 0;
        self.idx_1d = 0;
    }
}

impl ParamVisitor for SGDMomentum {
    fn visit_array2_with_grad(
        &mut self,
        param: &mut Array2<Float>,
        grad: &Array2<Float>,
    ) {
        if self.idx_2d >= self.velocities_2d.len() {
            self.velocities_2d.push(Array2::zeros(param.raw_dim()));
        }

        let velocity = &mut self.velocities_2d[self.idx_2d];

        azip!((p in param, g in grad, v in velocity) {
            *v = self.momentum * *v + *g;
            *p = *p - self.learning_rate * *v;
        });

        self.idx_2d += 1;
    }

    fn visit_array1_with_grad(
        &mut self,
        param: &mut Array1<Float>,
        grad: &Array1<Float>,
    ) {
        if self.idx_1d >= self.velocities_1d.len() {
            self.velocities_1d.push(Array1::zeros(param.raw_dim()));
        }

        let velocity = &mut self.velocities_1d[self.idx_1d];

        azip!((p in param, g in grad, v in velocity) {
            *v = self.momentum * *v + *g;
            *p = *p - self.learning_rate * *v;
        });

        self.idx_1d += 1;
    }
}
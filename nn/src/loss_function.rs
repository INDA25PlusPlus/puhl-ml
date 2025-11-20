use core::f64;

use ndarray::{Array2, Axis};

pub trait LossFunction {
    fn forward(&mut self, prediction: &Array2<f64>, target: &Array2<f64>) -> f64;
    fn backward(&self) -> Array2<f64>;
}

pub struct MSE {
    last_prediction: Option<Array2<f64>>,
    last_target: Option<Array2<f64>>,
}

impl MSE {
    pub fn new() -> Self {
        Self {
            last_prediction: None,
            last_target: None,
        }
    }
}

impl LossFunction for MSE {
    fn forward(&mut self, prediction: &Array2<f64>, target: &Array2<f64>) -> f64 {
        self.last_prediction = Some(prediction.clone());
        self.last_target = Some(target.clone());

        let diff = prediction - target;
        let num_elements = (diff.shape()[0] * diff.shape()[1]) as f64;

        diff.mapv(|x| x * x).sum() / num_elements
    }

    fn backward(&self) -> Array2<f64> {
        let prediction = self.last_prediction.as_ref()
            .expect("forward() must be called before backward()");
        let target = self.last_target.as_ref()
            .expect("forward() must be called before backward()");

        let num_elements = (prediction.shape()[0] * prediction.shape()[1]) as f64;

        (prediction - target) * (2.0 / num_elements)
    }
}


pub struct CrossEntropy {
    last_probs: Option<Array2<f64>>,
    last_target: Option<Array2<f64>>,
}

impl CrossEntropy {
    pub fn new() -> Self {
        Self {
            last_probs: None,
            last_target: None,
        }
    }
}

// Includes a Softmax layer
impl LossFunction for CrossEntropy {
    fn forward(&mut self, prediction: &Array2<f64>, target: &Array2<f64>) -> f64 {
        let batch_size = prediction.shape()[1] as f64;

        // Gives the max value of each vector prediction of each batch
        let max_prediction = prediction.map_axis(Axis(0), |col| {
            col.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        });

        // Keeps everything under 0 so exp doesn't explode in size
        let logits = prediction - &max_prediction;

        // Softmax
        let exp_logits = logits.exp();
        let sum_exp = exp_logits.sum_axis(Axis(0));
        let probs = &exp_logits / &sum_exp;

        // Save for backward
        self.last_probs = Some(probs.clone());
        self.last_target = Some(target.clone());

        // Cross entropy
        let log_probs = probs.mapv(|x| (x + 1e-15).ln());   // Add 1e-15; would be undefined otherwise if x was 0
        let total_loss = -(target * &log_probs).sum();

        return total_loss / batch_size;
    }

    fn backward(&self) -> Array2<f64> {
        let probs = self.last_probs.as_ref()
            .expect("forward() must be called before backward()");
        let target = self.last_target.as_ref()
            .expect("forward() must be called before backward()");

        let batch_size = probs.shape()[1] as f64;

        (probs - target) / batch_size
    }
}
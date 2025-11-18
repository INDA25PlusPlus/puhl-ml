use ndarray::Array2;

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
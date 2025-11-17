use super::visitor::ParamVisitor;

pub trait Layer {
    type Input;
    type Output;

    fn forward(&mut self, input: &Self::Input) -> Self::Output;
    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input;
}


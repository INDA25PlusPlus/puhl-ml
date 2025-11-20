use mnist::*;
use ndarray::prelude::*;
use nn::{
    activation_layers::ReLU,
    layer::Layer,
    param_layers::LinearLayer,
    visitor::{ParamVisitor, Parameterized},
    loss_function::{LossFunction, MSE},
    optimizer::SGD,
};

// Uses simple linear layers, ReLU and MSE
struct NetworkBasic {
    pub layer1: LinearLayer,
    pub relu: ReLU,
    pub layer2: LinearLayer,
}

impl NetworkBasic {
    pub fn new() -> Self {
        Self {
            layer1: LinearLayer::new(128, 784),
            relu: ReLU::new(),
            layer2: LinearLayer::new(10, 128),
        }
    }
}

impl Layer for NetworkBasic {
    type Input = Array2<f64>;
    type Output = Array2<f64>;

    fn forward(&mut self, input: &Self::Input) -> Self::Output {
        let x = self.layer1.forward(input);
        let x = self.relu.forward(&x);
        self.layer2.forward(&x)
    }

    fn backward(&mut self, grad_output: &Self::Output) -> Self::Input {
        let x = self.layer2.backward(grad_output);
        let x = self.relu.backward(&x);
        self.layer1.backward(&x)
    }
}

impl Parameterized for NetworkBasic {
    fn visit_params<V: ParamVisitor>(&mut self, visitor: &mut V) {
        self.layer1.visit_params(visitor);
        self.layer2.visit_params(visitor);
    }

    fn zero_grad(&mut self) {
        self.layer1.zero_grad();
        self.layer2.zero_grad();
    }
}

// Might come in handy later
fn _print_digit(image: ArrayView2<f32>, label: u8) {
    println!("\n=== Digit: {} ===\n", label);

    // Display using ASCII characters based on pixel intensity
    for row in 0..28 {
        for col in 0..28 {
            let pixel = image[[row, col]];
            let ch = if pixel > 0.75 {
                '@'
            } else if pixel > 0.5 {
                '#'
            } else if pixel > 0.25 {
                '+'
            } else if pixel > 0.1 {
                '.'
            } else {
                ' '
            };
            print!("{}", ch);
        }
        println!();
    }
    println!();
}

fn one_hot_encode(labels: &[u8], num_classes: usize) -> Array2<f64> {
    let n = labels.len();
    let mut encoded = Array2::zeros((num_classes, n));
    for (i, &label) in labels.iter().enumerate() {
        encoded[[label as usize, i]] = 1.0;
    }
    encoded
}

fn train_network(
    network: &mut NetworkBasic,
    train_images: &Array2<f64>,
    train_labels: &Array2<f64>,
    test_images: &Array2<f64>,
    test_labels: &[u8],
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
) {
    let mut loss_fn = MSE::new();
    let mut optimizer = SGD::new(learning_rate);
    let num_samples = train_images.shape()[1];

    println!("\nStarting training...");
    println!("Network: 784 -> 128 -> 10");
    println!("Epochs: {}, Batch size: {}, Learning rate: {}\n", epochs, batch_size, learning_rate);

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        // Training
        for batch_start in (0..num_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_samples);
            let batch_images = train_images.slice(s![.., batch_start..batch_end]);
            let batch_labels = train_labels.slice(s![.., batch_start..batch_end]);

            // Forward pass
            let output = network.forward(&batch_images.to_owned());
            let loss = loss_fn.forward(&output, &batch_labels.to_owned());
            total_loss += loss;
            num_batches += 1;

            // Backward pass
            let grad = loss_fn.backward();
            network.backward(&grad);

            // Update weights
            network.visit_params(&mut optimizer);
            network.zero_grad();
        }

        let avg_loss = total_loss / num_batches as f64;

        // Calculate accuracy every epoch
        let accuracy = calculate_accuracy(network, test_images, test_labels);

        println!("Epoch {}/{} - Loss: {:.6} - Test Accuracy: {:.2}%",
                 epoch + 1, epochs, avg_loss, accuracy * 100.0);
    }

    println!("\nTraining complete!");
}

fn calculate_accuracy(
    network: &mut NetworkBasic,
    test_images: &Array2<f64>,
    test_labels: &[u8],
) -> f64 {
    let output = network.forward(test_images);
    let predictions = output.map_axis(Axis(0), |col| {
        col.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap()
    });

    let correct = predictions.iter()
        .zip(test_labels.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();

    correct as f64 / test_labels.len() as f64
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_images_flat: Vec<f64> = trn_img.iter().map(|&x| x as f64 / 256.0).collect();
    let train_images = Array2::from_shape_vec((50_000, 784), train_images_flat)
        .expect("Error converting training images to Array2")
        .t()
        .to_owned();

    let test_images_flat: Vec<f64> = tst_img.iter().map(|&x| x as f64 / 256.0).collect();
    let test_images = Array2::from_shape_vec((10_000, 784), test_images_flat)
        .expect("Error converting test images to Array2")
        .t()
        .to_owned();

    let train_labels_encoded = one_hot_encode(&trn_lbl, 10);

    let mut network = NetworkBasic::new();

    use nn::initializer::XavierInitializer;
    let mut initializer = XavierInitializer::new();
    network.visit_params(&mut initializer);

    train_network(
        &mut network,
        &train_images,
        &train_labels_encoded,
        &test_images,
        &tst_lbl,
        50,    // epochs
        64,    // batch_size
        0.1,   // learning_rate
    );

    println!("\n=== Final Test ===");
    let final_accuracy = calculate_accuracy(&mut network, &test_images, &tst_lbl);
    println!("Final Test Accuracy: {:.2}%", final_accuracy * 100.0);
}
mod conjugate_gradient;
mod gradient_descent;
mod newton_raphson;
mod step_size;

pub use conjugate_gradient::{Conjugacy, ConjugateGradient};
pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;
pub use step_size::{BarzilaiBorwein, BarzilaiBorweinStep, StepEstimator};

#[cfg(feature = "constitutive")]
pub(crate) use newton_raphson::{converged, limit_decrement};

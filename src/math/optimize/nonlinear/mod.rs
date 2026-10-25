mod gradient_descent;
mod newton_raphson;

pub use gradient_descent::GradientDescent;
pub use newton_raphson::NewtonRaphson;

#[cfg(feature = "constitutive")]
pub(crate) use newton_raphson::{converged, limit_decrement};

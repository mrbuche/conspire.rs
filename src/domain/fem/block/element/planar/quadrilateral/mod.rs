#[cfg(feature = "autodiff")]
mod autodiff;

use crate::fem::block::element::Element;

pub type Quadrilateral = Element<2, 4, 4, 1>;

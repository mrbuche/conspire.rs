#[cfg(feature = "autodiff")]
mod autodiff;

use crate::fem::block::element::Element;

pub type Triangle = Element<2, 1, 3, 1>;

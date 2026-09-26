use crate::fem::block::element::solid::{
    hyperelastic::autodiff::autodiff_element,
    hyperviscoelastic::autodiff::autodiff_viscoelastic_element,
};

autodiff_element!(3, 18, 15, 2);
autodiff_viscoelastic_element!(18, 15, 2);

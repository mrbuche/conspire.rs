use crate::fem::block::element::solid::{
    hyperelastic::autodiff::autodiff_element,
    hyperviscoelastic::autodiff::autodiff_viscoelastic_element,
};

autodiff_element!(3, 8, 5, 1);
autodiff_viscoelastic_element!(8, 5, 1);

use crate::fem::block::element::solid::{
    hyperelastic::autodiff::autodiff_element,
    hyperviscoelastic::autodiff::autodiff_viscoelastic_element,
};

autodiff_element!(3, 6, 6, 1);
autodiff_viscoelastic_element!(6, 6, 1);

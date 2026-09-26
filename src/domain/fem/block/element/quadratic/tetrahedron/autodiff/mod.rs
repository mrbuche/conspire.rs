use crate::fem::block::element::solid::{
    hyperelastic::autodiff::autodiff_element,
    hyperviscoelastic::autodiff::autodiff_viscoelastic_element,
};

autodiff_element!(3, 4, 10, 2);
autodiff_viscoelastic_element!(4, 10, 2);

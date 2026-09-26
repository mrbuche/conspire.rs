use crate::fem::block::element::solid::hyperelastic::autodiff::{
    autodiff_element, autodiff_viscoelastic_element,
};

autodiff_element!(3, 1, 4, 1);
autodiff_viscoelastic_element!(1, 4, 1);

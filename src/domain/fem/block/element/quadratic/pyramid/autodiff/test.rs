use crate::fem::block::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::block::element::quadratic::Pyramid, 27, 13);
viscous_tests!(crate::fem::block::element::quadratic::Pyramid, 27, 13);

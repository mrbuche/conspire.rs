use crate::fem::block::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::block::element::linear::Pyramid, 8, 5);
viscous_tests!(crate::fem::block::element::linear::Pyramid, 8, 5);

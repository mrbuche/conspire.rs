use crate::fem::block::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::block::element::linear::Tetrahedron, 1, 4);
viscous_tests!(crate::fem::block::element::linear::Tetrahedron, 1, 4);

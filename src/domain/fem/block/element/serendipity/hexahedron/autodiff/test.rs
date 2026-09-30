use crate::fem::block::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::block::element::serendipity::Hexahedron, 27, 20);
viscous_tests!(crate::fem::block::element::serendipity::Hexahedron, 27, 20);

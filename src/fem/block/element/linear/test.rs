macro_rules! test_linear_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_elements!($element);
        use crate::math::TensorArray;
        crate::fem::block::element::test::test_finite_element!($element);
    };
}
pub(crate) use test_linear_element;

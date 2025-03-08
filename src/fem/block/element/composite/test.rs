macro_rules! test_composite_element {
    ($element: ident) => {
        crate::fem::block::element::test::setup_for_elements!($element);
        crate::fem::block::element::test::test_finite_element!($element);
    };
}
pub(crate) use test_composite_element;

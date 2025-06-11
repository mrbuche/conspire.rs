use super::{
    super::test::{TestError, assert_eq},
    Interpolate1D, LinearInterpolation, TensorVec, Vector,
};

#[test]
fn line() -> Result<(), TestError> {
    let x = Vector::new(&[0.324, 0.745]);
    let xp = Vector::new(&[0.0, 1.0]);
    let linear_function = |x: &Vector| x * 2.3;
    let fp = linear_function(&xp);
    let f = LinearInterpolation::interpolate_1d(&x, &xp, &fp);
    assert_eq(&linear_function(&x), &f)
}

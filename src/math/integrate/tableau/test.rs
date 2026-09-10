use super::{ButcherTableau, EmbeddedTableau};
use crate::math::integrate::ode::explicit::variable_step::{
    bogacki_shampine, dormand_prince, verner_8, verner_9,
};

const TOL_LOW_ORDER: f64 = 1e-12;
const TOL_HIGH_ORDER: f64 = 1e-11;

fn check_butcher<T: ButcherTableau>(tol: f64) {
    assert_eq!(T::A.len(), T::STAGES);
    assert_eq!(T::C.len(), T::STAGES);
    assert_eq!(T::B.len(), T::STAGES);
    assert!(T::ORDER > 0.0);
    assert_eq!(T::C[0], 0.0);
    T::A.iter().enumerate().for_each(|(i, row)| {
        assert_eq!(row.len(), i, "row {i} has the wrong length");
        let row_sum: f64 = row.iter().sum();
        assert!(
            (row_sum - T::C[i]).abs() < tol,
            "row {i} sums to {row_sum}, expected c = {}",
            T::C[i]
        );
    });
    let b_sum: f64 = T::B.iter().sum();
    assert!(
        (b_sum - 1.0).abs() < tol,
        "propagating weights sum to {b_sum}"
    );
}

fn check_embedded<T: EmbeddedTableau>(tol: f64) {
    check_butcher::<T>(tol);
    assert_eq!(T::D.len(), T::STAGES);
    let d_sum: f64 = T::D.iter().sum();
    assert!(
        d_sum.abs() < tol,
        "error weights sum to {d_sum}, expected 0"
    );
    if T::FSAL {
        assert_eq!(T::C[T::STAGES - 1], 1.0);
        T::A[T::STAGES - 1]
            .iter()
            .zip(T::B.iter())
            .for_each(|(a, b)| assert_eq!(a, b, "final row must equal the propagating weights"));
    }
}

#[test]
fn bogacki_shampine() {
    const { assert!(bogacki_shampine::Tableau::FSAL) };
    check_embedded::<bogacki_shampine::Tableau>(TOL_LOW_ORDER);
}

#[test]
fn dormand_prince() {
    const { assert!(dormand_prince::Tableau::FSAL) };
    check_embedded::<dormand_prince::Tableau>(TOL_LOW_ORDER);
}

#[test]
fn verner_8() {
    const { assert!(!verner_8::Tableau::FSAL) };
    check_embedded::<verner_8::Tableau>(TOL_HIGH_ORDER);
}

#[test]
fn verner_9() {
    const { assert!(!verner_9::Tableau::FSAL) };
    check_embedded::<verner_9::Tableau>(TOL_HIGH_ORDER);
}

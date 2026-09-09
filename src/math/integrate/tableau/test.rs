use super::{BogackiShampine32, ButcherTableau, DormandPrince54, EmbeddedTableau};

const TOL: f64 = 1e-12;

fn check_butcher<T: ButcherTableau>() {
    assert_eq!(T::A.len(), T::STAGES);
    assert_eq!(T::C.len(), T::STAGES);
    assert_eq!(T::B.len(), T::STAGES);
    assert!(T::ORDER > 0.0);
    assert_eq!(T::C[0], 0.0);
    T::A.iter().enumerate().for_each(|(i, row)| {
        assert_eq!(row.len(), i, "row {i} has the wrong length");
        let row_sum: f64 = row.iter().sum();
        assert!(
            (row_sum - T::C[i]).abs() < TOL,
            "row {i} sums to {row_sum}, expected c = {}",
            T::C[i]
        );
    });
    let b_sum: f64 = T::B.iter().sum();
    assert!(
        (b_sum - 1.0).abs() < TOL,
        "propagating weights sum to {b_sum}"
    );
}

fn check_embedded<T: EmbeddedTableau>() {
    check_butcher::<T>();
    assert_eq!(T::D.len(), T::STAGES);
    let d_sum: f64 = T::D.iter().sum();
    assert!(
        d_sum.abs() < TOL,
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
    const { assert!(BogackiShampine32::FSAL) };
    check_embedded::<BogackiShampine32>();
}

#[test]
fn dormand_prince() {
    const { assert!(DormandPrince54::FSAL) };
    check_embedded::<DormandPrince54>();
}

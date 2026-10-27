use super::{Jacobi, Precondition, Scaled};
use crate::{
    math::{Current, Quantity, QuantityVector, Tensor, TensorRank1Vec},
    units::{Force, ForcePerLength, Length, ReciprocalForcePerLength},
};

type Forces = TensorRank1Vec<3, Current, Force>;
type Displacements = TensorRank1Vec<3, Current, Length>;

fn forces() -> Forces {
    vec![[2.0, 4.0, 6.0], [3.0, 9.0, 12.0]].into()
}

fn entries(displacements: &Displacements) -> Vec<f64> {
    displacements
        .iter()
        .flat_map(|node| (0..3).map(|i| node[i].value()))
        .collect()
}

#[test]
fn a_multiple_of_the_identity_takes_force_to_length() {
    let displacements: Displacements =
        Scaled(Quantity::<ReciprocalForcePerLength>::new(0.5)).apply(&forces());
    assert_eq!(entries(&displacements), [1.0, 2.0, 3.0, 1.5, 4.5, 6.0])
}

#[test]
fn a_diagonal_divides_each_node_by_its_entry() {
    let diagonal: QuantityVector<ForcePerLength> = [2.0, 3.0]
        .into_iter()
        .map(Quantity::<ForcePerLength>::new)
        .collect();
    let displacements: Displacements = Jacobi(diagonal).apply(&forces());
    assert_eq!(entries(&displacements), [1.0, 2.0, 3.0, 1.0, 3.0, 4.0])
}

#[test]
fn a_diagonal_divides_scalar_items_too() {
    let diagonal: QuantityVector<ForcePerLength> = [2.0, 4.0]
        .into_iter()
        .map(Quantity::<ForcePerLength>::new)
        .collect();
    let loads: QuantityVector<Force> = [6.0, 8.0].into_iter().map(Quantity::new).collect();
    let lengths: QuantityVector<Length> = Jacobi(diagonal).apply(&loads);
    assert_eq!(
        lengths
            .iter()
            .map(|length| length.value())
            .collect::<Vec<_>>(),
        [3.0, 2.0]
    )
}

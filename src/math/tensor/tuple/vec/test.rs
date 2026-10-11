use crate::math::{
    Current, Jacobian, Matrix, Quantity, Solution, Tensor, TensorRank1, TensorTupleVec, Vector,
};

type Field = TensorTupleVec<TensorRank1<3, Current>, Quantity>;

fn vector() -> Vector {
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].into()
}

fn field() -> Field {
    vector().into()
}

#[test]
fn from_vector_is_node_major() {
    let field = field();
    assert_eq!(field.len(), 2);
    assert_eq!(
        (
            field[0].0[0].value(),
            field[0].0[2].value(),
            field[0].1.value()
        ),
        (0.0, 2.0, 3.0)
    );
    assert_eq!(
        (
            field[1].0[0].value(),
            field[1].0[2].value(),
            field[1].1.value()
        ),
        (4.0, 6.0, 7.0)
    )
}

#[test]
fn fill_into_undoes_from_vector() {
    let mut filled = Vector::zero(8);
    field().fill_into(&mut filled);
    assert_eq!(filled, vector())
}

#[test]
fn fill_into_chained_appends() {
    let mut filled = Vector::zero(10);
    field().fill_into_chained(Vector::from([8.0, 9.0]), &mut filled);
    assert_eq!(
        filled,
        Vector::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    )
}

#[test]
fn retain_from_keeps_the_marked_entries() {
    let retained = [true, false, true, true, false, true, false, true];
    assert_eq!(
        field().retain_from(&retained),
        Vector::from([0.0, 2.0, 3.0, 5.0, 7.0])
    )
}

#[test]
fn decrement_from_retained_touches_only_the_marked_entries() {
    let retained = [true, false, true, true, false, true, false, true];
    let mut field = field();
    field.decrement_from_retained(&retained, &Vector::from([10.0, 10.0, 10.0, 10.0, 10.0]));
    let mut filled = Vector::zero(8);
    field.fill_into(&mut filled);
    assert_eq!(
        filled,
        Vector::from([-10.0, 1.0, -8.0, -7.0, 4.0, -5.0, 6.0, -3.0])
    )
}

#[test]
fn decrement_from_matches_the_flat_vector() {
    let mut field = field();
    field.decrement_from(&vector());
    assert!(field.iter().all(|entry| entry.1.value() == 0.0))
}

#[test]
fn zero_out_clears_the_listed_entries() {
    let mut field = field();
    field.zero_out(&[1, 3, 4]);
    let mut filled = Vector::zero(8);
    field.fill_into(&mut filled);
    assert_eq!(
        filled,
        Vector::from([0.0, 0.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0])
    )
}

#[test]
fn matrix_times_the_field_is_the_flat_product() {
    let mut matrix = Matrix::zero(2, 8);
    (0..8).for_each(|j| {
        matrix[0][j] = 1.0;
        matrix[1][j] = j as f64
    });
    assert_eq!(&matrix * &field(), Vector::from([28.0, 140.0]))
}

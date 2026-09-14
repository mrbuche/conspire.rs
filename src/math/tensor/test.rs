use crate::math::{
    Current, HessianBlock, Quantity, Reference, SquareMatrix, Tensor, TensorArray, TensorRank1,
    TensorRank2, TensorRank4, Transposed,
};
use crate::units::{Dimensionless, Rate, Stress, Viscosity};

type Deformation = TensorRank2<3, Current, Reference, Dimensionless>;
type Rates = TensorRank2<3, Reference, Reference, Rate>;
type Viscosities = TensorRank2<3, Current, Reference, Viscosity>;
type Stresses = TensorRank2<3, Current, Reference, Stress>;

#[test]
fn a_unit_costs_no_space() {
    assert_eq!(size_of::<Stresses>(), size_of::<Deformation>());
}

#[test]
fn same_units_add() {
    let sum = Stresses::zero() + Stresses::zero();
    assert!(sum.is_zero())
}

#[test]
fn multiplication_combines_the_units() {
    // Viscosity * Rate = Stress, resolved when this compiles.
    let stress: Stresses = Viscosities::zero() * Rates::zero();
    assert!(stress.is_zero())
}

#[test]
fn the_default_is_dimensionless() {
    let product =
        TensorRank2::<3, Current, Reference>::zero() * TensorRank2::<3, Reference, Current>::zero();
    assert!(product.is_zero())
}

#[test]
fn transposed_swaps_dimensions_and_indices() {
    let column = TensorRank1::<3, Current>::from([1.0, 2.0, 3.0]);
    let row = Transposed(TensorRank1::<3, Current>::from([1.0, 2.0, 3.0]));
    assert_eq!((column.height(), column.width()), (3, 1));
    assert_eq!((row.height(), row.width()), (1, 3));
    assert_eq!(row.entry(0, 2), column.entry(2, 0));
}

#[test]
fn mixed_rank_blocks_assemble_into_one_kkt_matrix() {
    let kuu = TensorRank4::<3, Current, Reference, Current, Reference, Stress>::identity();
    let kuv =
        TensorRank1::<9, Current, Stress>::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let kvu = Transposed(TensorRank1::<9, Reference>::from([
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    ]));
    let kvv = Quantity::<Dimensionless>::new(100.0);
    assert_eq!((kuu.height(), kuu.width()), (9, 9));
    assert_eq!((kuv.height(), kuv.width()), (9, 1));
    assert_eq!((kvu.height(), kvu.width()), (1, 9));
    assert_eq!((kvv.height(), kvv.width()), (1, 1));
    let mut matrix = SquareMatrix::zero(10);
    kuu.fill_into_block(&mut matrix, 0, 0);
    (0..9).for_each(|i| (0..9).for_each(|j| assert_eq!(matrix[i][j], kuu.entry(i, j))));
    kuv.fill_into_block(&mut matrix, 0, 9);
    kvu.fill_into_block(&mut matrix, 9, 0);
    kvv.fill_into_block(&mut matrix, 9, 9);
    (0..9).for_each(|i| {
        assert_eq!(matrix[i][9], (i + 1) as f64);
        assert_eq!(matrix[9][i], 10.0 * (i + 1) as f64);
        (0..9).for_each(|j| assert_eq!(matrix[i][j], kuu.entry(i, j)));
    });
    assert_eq!(matrix[9][9], 100.0);
}

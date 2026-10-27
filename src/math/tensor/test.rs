use crate::math::{
    Current, HessianBlock, Quantity, Reference, SquareMatrix, Tensor, TensorArray, TensorRank1,
    TensorRank2, TensorRank4, Transposed,
};
use crate::units::Force;
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

mod erase {
    use crate::math::{Current, QuantityVector, Tensor, TensorRank1Vec};
    use crate::units::Force;
    #[test]
    fn an_owned_vector_is_erased_without_copying() {
        let vector = TensorRank1Vec::<3, Current, Force>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let pointer = vector.as_ptr() as *const u8;
        let erased = vector.into_erased();
        assert_eq!(erased.as_ptr() as *const u8, pointer);
        assert_eq!(erased.len(), 2);
        assert_eq!(erased[1][2], 6.0);
    }
    #[test]
    fn erased_quantities_keep_their_values() {
        let mut vector = QuantityVector::<Force>::zero(3);
        vector
            .iter_mut()
            .enumerate()
            .for_each(|(i, quantity)| *quantity = Force::newtons(i as f64 + 0.5));
        let erased = vector.into_erased();
        (0..3).for_each(|i| assert_eq!(erased[i], i as f64 + 0.5));
    }
}

mod flat_entries {
    use crate::{
        math::{
            Current, Quantity, QuantityVector, Scalar, Solution, TensorRank1, TensorRank1Vec,
            TensorTupleVec, Vector,
        },
        units::{Length, Temperature},
    };
    fn agree_with_decrementing<X>(size: usize)
    where
        X: Solution + Clone,
    {
        let mut solution = X::from(Vector::zero(size));
        (0..size).for_each(|i| solution.set_flat_entry(i, i as Scalar + 0.5));
        (0..size).for_each(|i| assert_eq!(solution.flat_entry(i), i as Scalar + 0.5));
        (0..size).for_each(|i| {
            let mut unit = Vector::zero(size);
            unit[i] = 1.0;
            let mut decremented = solution.clone();
            decremented.decrement_from(&unit);
            (0..size).for_each(|j| {
                let expected = solution.flat_entry(j) - if j == i { 1.0 } else { 0.0 };
                assert_eq!(decremented.flat_entry(j), expected, "index {i}, entry {j}")
            })
        });
    }
    #[test]
    fn a_vector() {
        agree_with_decrementing::<Vector>(7)
    }
    #[test]
    fn a_vector_of_quantities() {
        agree_with_decrementing::<QuantityVector<Length>>(7)
    }
    #[test]
    fn a_point() {
        agree_with_decrementing::<TensorRank1<3, Current, Length>>(3)
    }
    #[test]
    fn a_quantity() {
        agree_with_decrementing::<Quantity<Length>>(1)
    }
    #[test]
    fn coordinates() {
        agree_with_decrementing::<TensorRank1Vec<3, Current, Length>>(9)
    }
    #[test]
    fn coordinates_and_temperatures() {
        agree_with_decrementing::<
            TensorTupleVec<TensorRank1<3, Current, Length>, Quantity<Temperature>>,
        >(12)
    }
    #[test]
    fn the_temperature_follows_the_coordinates_of_its_node() {
        let mut solution =
            TensorTupleVec::<TensorRank1<3, Current, Length>, Quantity<Temperature>>::from(
                Vector::zero(8),
            );
        solution.set_flat_entry(7, 300.0);
        solution.set_flat_entry(5, 1.5);
        assert_eq!(solution[1].1.value(), 300.0);
        assert_eq!(solution[1].0[1].value(), 1.5);
        assert_eq!(solution[0].1.value(), 0.0);
    }
}

#[test]
fn a_contraction_over_a_length_of_the_unit_is_a_number() {
    let a = TensorRank1::<3, Current, Force>::from([1.0, 2.0, 3.0]);
    let b = TensorRank1::<3, Current, Force>::from([4.0, 5.0, 6.0]);
    let over = a.contraction_over(&b, Quantity::<Force>::new(2.0));
    assert_eq!(over.value(), 32.0 / 4.0);
    let _: Quantity<Dimensionless> = over;
}

use super::CscIncompleteLdl;
use crate::math::{Scalar, SquareMatrix, Tensor, Vector};

/// The five-point Laplacian of a square grid, the case incomplete factorization
/// was made for: the diagonals a row away from the main one fill in the whole
/// band between, and none of that fill is kept.
fn laplacian(side: usize) -> (usize, Vec<(usize, usize, Scalar)>) {
    let size = side * side;
    let mut entries = Vec::new();
    (0..side).for_each(|row| {
        (0..side).for_each(|column| {
            let here = row * side + column;
            entries.push((here, here, 4.0));
            if column > 0 {
                entries.push((here, here - 1, -1.0))
            }
            if row > 0 {
                entries.push((here, here - side, -1.0))
            }
        })
    });
    (size, entries)
}

fn dense(size: usize, entries: &[(usize, usize, Scalar)]) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(size);
    entries.iter().for_each(|&(i, j, value)| {
        matrix[i][j] += value;
        if i != j {
            matrix[j][i] += value
        }
    });
    matrix
}

/// What the factorization stands for, with the pivots taken as they came out
/// rather than by magnitude.
/// What the factorization stands for, put back in the terms the matrix was
/// given in: the factor is of the scaled matrix, so the scaling comes back out
/// on both sides.
fn product(factorization: &CscIncompleteLdl, size: usize) -> SquareMatrix {
    let mut product = SquareMatrix::zero(size);
    (0..size).for_each(|row| {
        (0..size).for_each(|column| {
            product[row][column] = (0..size)
                .map(|k| {
                    factorization.entry(row, k)
                        * factorization.pivot(k)
                        * factorization.entry(column, k)
                })
                .sum::<Scalar>()
                / (factorization.scale(row) * factorization.scale(column))
        })
    });
    product
}

/// A factor that drops nothing is the complete factorization, and stands in
/// for the inverse exactly whatever order pivoting chose to eliminate in.
///
/// A tridiagonal matrix has no fill to speak of only under the order its own
/// rows come in — every row's diagonal starts tied after scaling, so pivoting
/// by largest-available-diagonal reorders even this, and reordering a chain
/// does create fill. What is tested here is not that this matrix happens to
/// have none, but that giving the factorization room to keep whatever fill
/// pivoting does create is what makes it exact, regardless of the order.
#[test]
fn exact_where_nothing_is_dropped() {
    let size = 12;
    let entries: Vec<_> = (0..size)
        .flat_map(|i| {
            let mut row = vec![(i, i, 2.0)];
            if i > 0 {
                row.push((i, i - 1, -1.0))
            }
            row
        })
        .collect();
    let matrix = dense(size, &entries);
    let factorization = CscIncompleteLdl::with_fill(size, entries, size, 0.0).unwrap();
    let right_hand_side = Vector::from((0..size).map(|i| (i as Scalar).sin()).collect::<Vec<_>>());
    let solution = factorization.solve(&right_hand_side);
    let residual = (matrix * &solution - right_hand_side.clone())
        .norm()
        .value();
    assert!(residual < 1e-10 * right_hand_side.norm().value())
}

/// What defines the factorization is not how near its product comes to the
/// matrix, but where it is exact: on every position the matrix keeps, the
/// product agrees with it entry for entry. Everywhere else the product is
/// whatever the dropped fill would have cancelled, and is not asked about.
#[test]
fn agrees_with_the_matrix_on_its_own_pattern() {
    let (size, entries) = laplacian(7);
    let matrix = dense(size, &entries);
    let factorization = CscIncompleteLdl::new(size, entries).unwrap();
    let product = product(&factorization, size);
    let mut off_pattern = 0.0 as Scalar;
    (0..size).for_each(|row| {
        (0..size).for_each(|column| {
            if matrix[row][column] != 0.0 {
                assert!(
                    (product[row][column] - matrix[row][column]).abs() < 1e-12,
                    "({row}, {column}) kept but disagrees"
                )
            } else {
                off_pattern = off_pattern.max((product[row][column] - matrix[row][column]).abs())
            }
        })
    });
    assert!(off_pattern > 1e-6, "no fill was actually dropped")
}

/// A positive definite matrix has every pivot positive, so taking magnitudes
/// changes nothing and this is an incomplete Cholesky factorization by another
/// arrangement.
#[test]
fn keeps_every_pivot_where_the_matrix_is_definite() {
    let (size, entries) = laplacian(7);
    assert_eq!(
        CscIncompleteLdl::new(size, entries)
            .unwrap()
            .negative_pivots(),
        0
    )
}

/// An indefinite matrix is where this earns its keep. Nothing breaks down, the
/// factorization is still exact on the pattern, and what it is applied as is
/// positive definite even though what it factorized was not.
#[test]
fn factorizes_an_indefinite_matrix_and_applies_a_definite_one() {
    let (size, mut entries) = laplacian(6);
    //
    // Turning a diagonal entry over takes the matrix indefinite, which no
    // Cholesky factorization survives.
    //
    entries[0].2 = -4.0;
    let matrix = dense(size, &entries);
    assert!(matrix.clone().factorize_ldl().is_ok());
    let factorization = CscIncompleteLdl::new(size, entries).unwrap();
    assert!(factorization.negative_pivots() > 0);
    let product = product(&factorization, size);
    (0..size).for_each(|row| {
        (0..size).for_each(|column| {
            if matrix[row][column] != 0.0 {
                assert!((product[row][column] - matrix[row][column]).abs() < 1e-10)
            }
        })
    });
    //
    // What is solved against is the magnitudes, so every quadratic form of it
    // is positive — which is what the walk needs and what the matrix itself
    // cannot supply.
    //
    let mut applied = SquareMatrix::zero(size);
    (0..size).for_each(|column| {
        let mut unit = Vector::zero(size);
        unit[column] = 1.0;
        let solved = factorization.solve(&unit);
        (0..size).for_each(|row| applied[row][column] = solved[row])
    });
    assert!(applied.factorize_ldl().is_ok());
    (0..size).for_each(|column| {
        let mut unit = Vector::zero(size);
        unit[column] = 1.0;
        let solved = factorization.solve(&unit);
        assert!(
            solved[column] > 0.0,
            "quadratic form at {column} not positive"
        )
    })
}

/// Positions handed over more than once are summed, an assembled triangle being
/// as acceptable as a merged one.
#[test]
fn sums_repeated_positions() {
    let once = CscIncompleteLdl::new(2, [(0, 0, 4.0), (1, 0, 1.0), (1, 1, 3.0)]).unwrap();
    let twice = CscIncompleteLdl::new(
        2,
        [
            (0, 0, 1.0),
            (0, 0, 3.0),
            (1, 0, 0.25),
            (1, 0, 0.75),
            (1, 1, 3.0),
        ],
    )
    .unwrap();
    let right_hand_side = Vector::from([1.0, 2.0]);
    assert_eq!(once.solve(&right_hand_side), twice.solve(&right_hand_side))
}

/// A row with no diagonal has no pivot, and there is nothing to be done with
/// such a matrix but say so.
#[test]
fn refuses_a_missing_diagonal() {
    assert!(CscIncompleteLdl::new(2, [(0, 0, 1.0), (1, 0, 1.0)]).is_none())
}

/// A stress test for the invariant at a scale a small matrix cannot exercise:
/// a heavily indefinite matrix, fill kept rather than dropped, still comes out
/// exact on its own pattern and still applies as something positive definite.
///
/// This does not by itself demonstrate that pivoting changed anything — a
/// Laplacian-derived matrix never drives a scaled diagonal near `SAFE`, so
/// elimination stays in the matrix's own order here regardless. What it does
/// catch is any bug in the permutation bookkeeping the pivoting rewrite
/// introduced (`permutation`, `position`, step-indexed storage) that a 6-row
/// matrix is too small to expose.
#[test]
fn stands_up_to_scale_and_heavy_indefiniteness() {
    let (size, mut entries) = laplacian(10);
    (0..size).step_by(3).for_each(|i| {
        let row = entries
            .iter()
            .position(|&(r, c, _)| r == i && c == i)
            .unwrap();
        entries[row].2 = -4.0;
    });
    let matrix = dense(size, &entries);
    let factorization = CscIncompleteLdl::with_fill(size, entries, 20, 0.0).unwrap();
    assert!(factorization.negative_pivots() > size / 4);
    assert!(factorization.growth().0 < 10.0);
    let product = product(&factorization, size);
    (0..size).for_each(|row| {
        (0..size).for_each(|column| {
            if matrix[row][column] != 0.0 {
                assert!((product[row][column] - matrix[row][column]).abs() < 1e-6)
            }
        })
    });
    let mut applied = SquareMatrix::zero(size);
    (0..size).for_each(|column| {
        let mut unit = Vector::zero(size);
        unit[column] = 1.0;
        let solved = factorization.solve(&unit);
        (0..size).for_each(|row| applied[row][column] = solved[row])
    });
    assert!(applied.factorize_ldl().is_ok())
}

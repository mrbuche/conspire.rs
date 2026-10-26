use super::{Krylov, KrylovMethod};
use crate::math::{SquareMatrix, Vector, assert::Assert, optimize::Preconditioning};

fn krylov() -> Krylov {
    Krylov {
        rel_tol: 1e-15,
        ..Default::default()
    }
}

fn minres() -> Krylov {
    Krylov {
        method: KrylovMethod::Minres,
        rel_tol: 1e-15,
        ..Default::default()
    }
}

fn coupled() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 4.0;
    matrix[1][1] = 3.0;
    matrix[2][2] = 5.0;
    matrix[0][1] = 1.0;
    matrix[1][0] = 1.0;
    matrix[1][2] = 2.0;
    matrix[2][1] = 2.0;
    matrix
}

fn indefinite() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 2.0;
    matrix[1][1] = 3.0;
    matrix[0][2] = 1.0;
    matrix[2][0] = 1.0;
    matrix[1][2] = 1.0;
    matrix[2][1] = 1.0;
    matrix
}

fn right_hand_side() -> Vector {
    [1.0, 2.0, 3.0].into_iter().collect()
}

fn apply(matrix: &SquareMatrix) -> impl FnMut(&Vector) -> Vector + '_ {
    |direction| matrix.clone() * direction
}

#[test]
fn matches_the_direct_solve() {
    let matrix = coupled();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = krylov()
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn diagonal_preconditioner_also_matches() {
    let matrix = coupled();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let diagonal: Vector = (0..3).map(|i| matrix[i][i]).collect();
    let solution = krylov()
        .solve(apply(&matrix), Preconditioning::Diagonal(diagonal), &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn refuses_a_nonpositive_curvature() {
    let matrix = indefinite();
    let rhs = right_hand_side();
    assert!(
        krylov()
            .solve(apply(&matrix), Preconditioning::None, &rhs)
            .is_err()
    );
}

#[test]
fn minres_serves_the_indefinite_system() {
    let matrix = indefinite();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = minres()
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

fn gmres(restart: usize) -> Krylov {
    Krylov {
        method: KrylovMethod::Gmres(restart),
        rel_tol: 1e-14,
        ..Default::default()
    }
}

fn nonsymmetric() -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(3);
    matrix[0][0] = 4.0;
    matrix[1][1] = 3.0;
    matrix[2][2] = 5.0;
    matrix[0][1] = 1.0;
    matrix[1][0] = -2.0;
    matrix[1][2] = 2.0;
    matrix[2][1] = 0.5;
    matrix[0][2] = -1.0;
    matrix
}

#[test]
fn gmres_serves_the_nonsymmetric_system() {
    let matrix = nonsymmetric();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = gmres(30)
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn gmres_restarts() {
    let matrix = nonsymmetric();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = gmres(2)
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn gmres_diagonal_preconditioner_matches() {
    let matrix = nonsymmetric();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let diagonal: Vector = (0..3).map(|i| matrix[i][i]).collect();
    let solution = gmres(30)
        .solve(apply(&matrix), Preconditioning::Diagonal(diagonal), &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn gmres_serves_the_indefinite_system() {
    let matrix = indefinite();
    let rhs = right_hand_side();
    let expected = matrix.clone().solve_lu(&rhs).unwrap();
    let solution = gmres(30)
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default()
        .eq_within_tols(&solution, &expected)
        .unwrap();
}

#[test]
fn gmres_zero_right_hand_side() {
    let matrix = nonsymmetric();
    let rhs = Vector::zero(3);
    let solution = gmres(30)
        .solve(apply(&matrix), Preconditioning::None, &rhs)
        .unwrap();
    Assert::default().eq_within_tols(&solution, &rhs).unwrap();
}

mod typed {
    use super::super::KrylovError;
    use super::*;
    use crate::{
        math::{Current, Tensor, TensorRank1},
        units::{Force, Length},
    };
    type Displacement = TensorRank1<3, Current, Length>;
    type Load = TensorRank1<3, Current, Force>;
    fn load() -> Load {
        [1.0, 2.0, 3.0].into()
    }
    fn stiffness(matrix: SquareMatrix) -> impl FnMut(&Displacement) -> Load {
        move |displacement| {
            let applied = |row: usize| -> f64 {
                (0..3)
                    .map(|column| matrix[row][column] * displacement[column].value())
                    .sum()
            };
            Load::from([applied(0), applied(1), applied(2)])
        }
    }
    fn jacobi(matrix: SquareMatrix) -> impl Fn(&Load) -> Displacement {
        move |load| {
            Displacement::from([
                load[0].value() / matrix[0][0],
                load[1].value() / matrix[1][1],
                load[2].value() / matrix[2][2],
            ])
        }
    }
    fn flat(displacement: &Displacement) -> Vector {
        (0..3).map(|i| displacement[i].value()).collect()
    }
    #[test]
    fn matches_the_direct_solve() {
        let expected = coupled().solve_lu(&right_hand_side()).unwrap();
        let solution = krylov()
            .conjugate_gradients(stiffness(coupled()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(flat(&solution), &expected)
            .unwrap();
    }
    #[test]
    fn matches_the_solve_without_units() {
        let matrix = coupled();
        let bare = krylov()
            .solve(apply(&matrix), Preconditioning::None, &right_hand_side())
            .unwrap();
        let typed = krylov()
            .conjugate_gradients(
                stiffness(coupled()),
                |load: &Load| {
                    Displacement::from([load[0].value(), load[1].value(), load[2].value()])
                },
                &load(),
            )
            .unwrap();
        Assert::default()
            .eq_within_tols(flat(&typed), &bare)
            .unwrap();
    }
    #[test]
    fn refuses_a_nonpositive_curvature() {
        assert!(matches!(
            krylov().conjugate_gradients(stiffness(indefinite()), jacobi(coupled()), &load()),
            Err(KrylovError::NotPositiveDefinite(_))
        ))
    }
    #[test]
    fn a_zero_right_hand_side_gives_a_zero_solution() {
        let solution = krylov()
            .conjugate_gradients(
                stiffness(coupled()),
                jacobi(coupled()),
                &Load::from([0.0; 3]),
            )
            .unwrap();
        assert!(flat(&solution).iter().all(|&entry| entry == 0.0))
    }
    #[test]
    fn running_out_of_steps_reports_the_relative_residual() {
        let one_step = Krylov {
            max_steps: 1,
            rel_tol: 1e-15,
            ..Default::default()
        };
        match one_step.conjugate_gradients(stiffness(coupled()), jacobi(coupled()), &load()) {
            Err(KrylovError::MaximumStepsReached(1, relative)) => {
                assert!(relative > 0.0 && relative < 1.0)
            }
            _ => panic!("expected the steps to run out"),
        }
    }
}

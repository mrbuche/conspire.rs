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
            .solve(stiffness(coupled()), jacobi(coupled()), &load())
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
            .solve(
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
            krylov().solve(stiffness(indefinite()), jacobi(coupled()), &load()),
            Err(KrylovError::NotPositiveDefinite(_))
        ))
    }
    #[test]
    fn a_zero_right_hand_side_gives_a_zero_solution() {
        let solution = krylov()
            .solve(
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
        match one_step.solve(stiffness(coupled()), jacobi(coupled()), &load()) {
            Err(KrylovError::MaximumStepsReached(1, relative)) => {
                assert!(relative > 0.0 && relative < 1.0)
            }
            _ => panic!("expected the steps to run out"),
        }
    }
    fn scaled_load(scale: f64) -> Load {
        Load::from([1.0 * scale, 2.0 * scale, 3.0 * scale])
    }
    fn scaled_expected(matrix: &SquareMatrix, scale: f64) -> Vector {
        matrix
            .clone()
            .solve_lu(
                &[1.0 * scale, 2.0 * scale, 3.0 * scale]
                    .into_iter()
                    .collect(),
            )
            .unwrap()
    }
    #[test]
    fn minres_serves_the_indefinite_system() {
        let solution = minres()
            .solve(stiffness(indefinite()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &indefinite().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn minres_does_not_depend_on_the_size_of_the_load() {
        for scale in [1e-8, 1e-3, 1e5, 1e9] {
            let solution = minres()
                .solve(
                    stiffness(indefinite()),
                    jacobi(coupled()),
                    &scaled_load(scale),
                )
                .unwrap();
            let expected = scaled_expected(&indefinite(), scale);
            let error = (0..3)
                .map(|i| (solution[i].value() - expected[i]).abs())
                .fold(0.0_f64, f64::max);
            assert!(error <= 1e-9 * scale, "off by {error:e} at {scale:e}")
        }
    }
    #[test]
    fn minres_refuses_a_preconditioner_that_is_not_positive_definite() {
        let negative = |load: &Load| -> Displacement {
            Displacement::from([-load[0].value(), -load[1].value(), -load[2].value()])
        };
        assert!(matches!(
            minres().solve(stiffness(coupled()), negative, &load()),
            Err(KrylovError::PreconditionerNotPositiveDefinite(_))
        ))
    }
    #[test]
    fn minres_gives_a_zero_solution_for_a_zero_load() {
        let solution = minres()
            .solve(
                stiffness(coupled()),
                jacobi(coupled()),
                &Load::from([0.0; 3]),
            )
            .unwrap();
        assert!(flat(&solution).iter().all(|&entry| entry == 0.0))
    }
    #[test]
    fn gmres_serves_the_nonsymmetric_system() {
        let solution = gmres(30)
            .solve(stiffness(nonsymmetric()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &nonsymmetric().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn gmres_restarts() {
        let solution = gmres(2)
            .solve(stiffness(nonsymmetric()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &nonsymmetric().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn gmres_does_not_depend_on_the_size_of_the_load() {
        for scale in [1e-8, 1e-3, 1e5, 1e9] {
            let solution = gmres(30)
                .solve(
                    stiffness(nonsymmetric()),
                    jacobi(coupled()),
                    &scaled_load(scale),
                )
                .unwrap();
            let expected = scaled_expected(&nonsymmetric(), scale);
            let error = (0..3)
                .map(|i| (solution[i].value() - expected[i]).abs())
                .fold(0.0_f64, f64::max);
            assert!(error <= 1e-9 * scale, "off by {error:e} at {scale:e}")
        }
    }
    #[test]
    fn gmres_matches_the_solve_without_units() {
        let matrix = nonsymmetric();
        let bare = gmres(30)
            .solve(apply(&matrix), Preconditioning::None, &right_hand_side())
            .unwrap();
        let typed = gmres(30)
            .solve(
                stiffness(nonsymmetric()),
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
    fn gmres_gives_a_zero_solution_for_a_zero_load() {
        let solution = gmres(30)
            .solve(
                stiffness(coupled()),
                jacobi(coupled()),
                &Load::from([0.0; 3]),
            )
            .unwrap();
        assert!(flat(&solution).iter().all(|&entry| entry == 0.0))
    }
    #[test]
    fn gmres_running_out_of_steps_reports_the_relative_residual() {
        let short = Krylov {
            max_steps: 1,
            method: KrylovMethod::Gmres(30),
            rel_tol: 1e-15,
        };
        match short.solve(stiffness(nonsymmetric()), jacobi(coupled()), &load()) {
            Err(KrylovError::MaximumStepsReached(1, relative)) => {
                assert!(relative > 0.0 && relative < 1.0)
            }
            _ => panic!("expected the steps to run out"),
        }
    }
    #[test]
    fn gmres_converges_within_one_cycle() {
        let one_cycle = Krylov {
            max_steps: 3,
            method: KrylovMethod::Gmres(30),
            rel_tol: 1e-12,
        };
        let solution = one_cycle
            .solve(stiffness(nonsymmetric()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &nonsymmetric().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn gmres_converges_within_one_cycle_without_a_preconditioner() {
        let one_cycle = Krylov {
            max_steps: 3,
            method: KrylovMethod::Gmres(30),
            rel_tol: 1e-12,
        };
        let solution = one_cycle
            .solve(
                stiffness(nonsymmetric()),
                |load: &Load| {
                    Displacement::from([load[0].value(), load[1].value(), load[2].value()])
                },
                &load(),
            )
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &nonsymmetric().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn minres_converges_within_the_dimension() {
        let exact = Krylov {
            max_steps: 3,
            method: KrylovMethod::Minres,
            rel_tol: 1e-12,
        };
        let solution = exact
            .solve(stiffness(indefinite()), jacobi(coupled()), &load())
            .unwrap();
        Assert::default()
            .eq_within_tols(
                flat(&solution),
                &indefinite().solve_lu(&right_hand_side()).unwrap(),
            )
            .unwrap();
    }
    #[test]
    fn minres_stops_when_the_residual_cannot_be_shortened() {
        let nothing = |_: &Displacement| Load::from([0.0; 3]);
        assert!(matches!(
            minres().solve(nothing, jacobi(coupled()), &load()),
            Err(KrylovError::StoppedShortening(..))
        ))
    }
    #[test]
    fn minres_refuses_a_preconditioner_that_stops_being_positive_definite() {
        let calls = std::cell::Cell::new(0);
        let fading = move |load: &Load| {
            calls.set(calls.get() + 1);
            let sign = if calls.get() == 1 { 1.0 } else { -1.0 };
            Displacement::from([
                sign * load[0].value(),
                sign * load[1].value(),
                sign * load[2].value(),
            ])
        };
        assert!(matches!(
            minres().solve(stiffness(coupled()), fading, &load()),
            Err(KrylovError::PreconditionerNotPositiveDefinite(_))
        ))
    }
}

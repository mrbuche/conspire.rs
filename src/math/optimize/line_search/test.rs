use super::LineSearch;
use crate::math::Scalar;

/// Half a square, from an argument of one, stepped along a decrement of one.
///
/// The slope there is one, so the thresholds are the controls themselves, and
/// the step that lands on the minimum is one.
fn search(line_search: LineSearch, step_size: Scalar) -> Result<Scalar, String> {
    line_search
        .search(
            |x: &Scalar, _: Scalar| Ok(x.powi(2) / 2.0),
            |x: &Scalar| Ok(*x),
            &1.0,
            &1.0,
            &1.0,
            step_size,
        )
        .map_err(|error| format!("{error}"))
}

fn wolfe(strong: bool) -> LineSearch {
    LineSearch::Wolfe {
        control_1: 1e-3,
        control_2: 4e-1,
        cut_back: 5e-1,
        max_steps: 100,
        strong,
    }
}

#[test]
fn wolfe_curvature_condition_follows_the_flag() {
    assert_eq!(search(wolfe(false), 3.0).unwrap(), 1.5);
    assert_eq!(search(wolfe(true), 3.0).unwrap(), 0.75)
}

/// The curvature conditions accept an interval, and a step below it can only be
/// reached by lengthening.
///
/// The controls put the strong interval at `[0.6, 1.4]`, so a tenth of a step is
/// well below it and no amount of shortening would ever arrive. Doubling gets
/// there in three.
mod below_the_interval {
    use super::*;
    #[test]
    fn wolfe_grows_onto_it() {
        assert_eq!(search(wolfe(true), 0.1).unwrap(), 0.8);
        assert_eq!(search(wolfe(false), 0.1).unwrap(), 0.8)
    }
    #[test]
    fn goldstein_grows_onto_it() {
        assert_eq!(
            search(
                LineSearch::Goldstein {
                    control: 3e-1,
                    cut_back: 5e-1,
                    max_steps: 100,
                },
                0.1,
            )
            .unwrap(),
            0.8
        )
    }
    /// Sufficient decrease alone is met by every step short enough, so there is
    /// no interval to grow onto and the step offered is taken as it is.
    #[test]
    fn armijo_stays_put() {
        assert_eq!(
            search(
                LineSearch::Armijo {
                    control: 1e-3,
                    cut_back: 9e-1,
                    max_steps: 100,
                },
                0.1,
            )
            .unwrap(),
            0.1
        )
    }
}

/// Growing is bounded by the same budget as bisecting, so a condition that is
/// never met is reported rather than chased forever.
#[test]
fn unreachable_interval_reports_the_budget() {
    assert!(
        search(
            LineSearch::Wolfe {
                control_1: 1e-3,
                control_2: 4e-1,
                cut_back: 5e-1,
                max_steps: 3,
                strong: true,
            },
            1e-6,
        )
        .is_err()
    )
}

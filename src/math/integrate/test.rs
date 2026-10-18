use crate::math::{Quantity, Tensor, TensorRank0, integrate::fixed_step_times};
use crate::units::Time;
use std::array::from_fn;
pub const LENGTH: usize = 33;

pub fn zero_to_one<const W: usize>() -> [Quantity<Time>; W] {
    from_fn(|i| Quantity::new((i as TensorRank0) / ((W - 1) as TensorRank0)))
}

fn grid(t_0: f64, t_f: f64, dt: f64) -> Vec<f64> {
    fixed_step_times(
        Quantity::new(t_0),
        Quantity::new(t_f),
        Quantity::<Time>::new(dt),
    )
    .iter()
    .map(|time| time.value())
    .collect()
}

#[test]
fn a_span_that_rounds_up_to_one_more_step_does_not_repeat_its_final_time() {
    let times = grid(0.0, 2000.0 * 1e-6, 1e-6);
    let dt = std::hint::black_box(1e-6_f64);
    assert!((2000.0 * dt) / dt > 2000.0);
    assert_eq!(times.len(), 2001);
    assert!(times[2000] > times[1999]);
    assert_eq!(times[2000], 2000.0 * 1e-6);
}

#[test]
fn every_whole_number_of_steps_ends_each_step_once() {
    for dt in [1e-6, 1e-5, 1e-3, 0.003, 0.01, 0.1, 0.25, 1.0 / 3.0] {
        for start in [0.0, 1.5] {
            for steps in 1..=5000 {
                let times = grid(start, start + steps as f64 * dt, dt);
                assert_eq!(times.len(), steps + 1, "{steps} steps of {dt} from {start}");
                assert_eq!(times[0], start);
                times.windows(2).for_each(|pair| {
                    let step = pair[1] - pair[0];
                    assert!(step > 0.0 && step <= dt * (1.0 + 1e-9), "{step} of {dt}");
                });
            }
        }
    }
}

#[test]
fn a_span_that_is_not_whole_steps_ends_with_a_shorter_one() {
    assert_eq!(grid(0.0, 2.5, 1.0), [0.0, 1.0, 2.0, 2.5]);
    assert_eq!(grid(1.0, 1.25, 0.1), [1.0, 1.1, 1.2, 1.25]);
}

#[test]
fn a_step_longer_than_the_span_is_one_step() {
    assert_eq!(grid(0.0, 1.0, 5.0), [0.0, 1.0]);
}

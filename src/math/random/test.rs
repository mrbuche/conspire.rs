mod random_u8 {
    use crate::math::random::random_u8;
    #[test]
    fn u8_max() {
        random_u8(u8::MAX);
    }
    #[test]
    fn one() {
        assert!(random_u8(1) < 2)
    }
    #[test]
    fn zero() {
        assert_eq!(random_u8(0), 0)
    }
}

mod rng {
    use crate::math::random::Rng;
    #[test]
    fn known_sequence() {
        let mut rng = Rng::new(1);
        assert_eq!(rng.next_u64(), 5180492295206395165);
    }
    #[test]
    fn deterministic_in_seed() {
        let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
        let (a, b, c): (Vec<u64>, Vec<u64>, Vec<u64>) = (
            (0..8).map(|_| a.next_u64()).collect(),
            (0..8).map(|_| b.next_u64()).collect(),
            (0..8).map(|_| c.next_u64()).collect(),
        );
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
    #[test]
    fn zero_seed_is_not_stuck() {
        let mut rng = Rng::new(0);
        assert_ne!(rng.next_u64(), 0);
        assert_ne!(rng.next_u64(), rng.next_u64());
    }
    #[test]
    fn uniform_in_unit_interval() {
        let mut rng = Rng::new(3);
        let samples: Vec<f64> = (0..10_000).map(|_| rng.uniform()).collect();
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 0.5).abs() < 0.02);
    }
    #[test]
    fn shuffle_permutes_deterministically() {
        let original: Vec<usize> = (0..50).collect();
        let (mut a, mut b) = (original.clone(), original.clone());
        Rng::new(5).shuffle(&mut a);
        Rng::new(5).shuffle(&mut b);
        assert_eq!(a, b);
        assert_ne!(a, original);
        a.sort_unstable();
        assert_eq!(a, original);
    }
    #[test]
    fn shuffle_short_slices() {
        let mut empty: [u8; 0] = [];
        Rng::new(1).shuffle(&mut empty);
        let mut one = [9];
        Rng::new(1).shuffle(&mut one);
        assert_eq!(one, [9]);
    }
}

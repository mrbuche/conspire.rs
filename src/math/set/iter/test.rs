use crate::math::set::Set;

#[test]
fn unnumbered_falls_back_to_index() {
    let set = Set::from(vec!['a', 'b', 'c']);
    let pairs: Vec<(usize, &char)> = set.iter().collect();
    assert_eq!(pairs, vec![(0, &'a'), (1, &'b'), (2, &'c')]);
}

#[test]
fn numbered_uses_stored_numbers() {
    let set: Set<Vec<char>> = Set::from((vec!['a', 'b', 'c'], vec![10, 20, 30]));
    let pairs: Vec<(usize, &char)> = set.iter().collect();
    assert_eq!(pairs, vec![(10, &'a'), (20, &'b'), (30, &'c')]);
}

#[test]
fn into_iter_unnumbered() {
    let set = Set::from(vec!['a', 'b', 'c']);
    let pairs: Vec<(usize, char)> = set.into_iter().collect();
    assert_eq!(pairs, vec![(0, 'a'), (1, 'b'), (2, 'c')]);
}

#[test]
fn into_iter_numbered() {
    let set: Set<Vec<char>> = Set::from((vec!['a', 'b', 'c'], vec![10, 20, 30]));
    let pairs: Vec<(usize, char)> = set.into_iter().collect();
    assert_eq!(pairs, vec![(10, 'a'), (20, 'b'), (30, 'c')]);
}

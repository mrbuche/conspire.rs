use super::{FxHashMap, FxHashSet};

#[test]
fn map_inserts_and_reads() {
    let mut map: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    map.insert((1, 2), 10);
    map.insert((3, 4), 20);
    assert_eq!(map.get(&(1, 2)), Some(&10));
    assert_eq!(map.get(&(3, 4)), Some(&20));
    assert_eq!(map.get(&(2, 1)), None);
    assert_eq!(map.len(), 2);
}

#[test]
fn set_deduplicates() {
    let mut set: FxHashSet<usize> = FxHashSet::default();
    set.insert(7);
    set.insert(7);
    set.insert(9);
    assert_eq!(set.len(), 2);
    assert!(set.contains(&7));
    assert!(!set.contains(&8));
}

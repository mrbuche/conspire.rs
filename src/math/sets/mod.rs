use std::collections::{HashMap, hash_map::IntoValues};

struct DisjointSetUnion {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSetUnion {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.find(p);
        }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
    }
}

pub fn disjoint_set_union<T>(set_members: &[T], num_members: usize) -> IntoValues<usize, Vec<usize>>
where
    for<'a> &'a T: IntoIterator<Item = &'a usize>,
{
    let mut member_sets = vec![vec![]; num_members];
    set_members.iter().enumerate().for_each(|(set, members)| {
        members
            .into_iter()
            .for_each(|&member| member_sets[member].push(set))
    });
    let num_sets = set_members.len();
    let mut dsu = DisjointSetUnion::new(num_sets);
    member_sets
        .into_iter()
        .filter(|v| v.len() >= 2)
        .for_each(|sets| {
            let first = sets[0];
            sets[1..].iter().for_each(|&s| dsu.union(first, s))
        });
    let mut disjoint_sets: HashMap<usize, Vec<usize>> = HashMap::new();
    (0..num_sets).for_each(|s| disjoint_sets.entry(dsu.find(s)).or_default().push(s));
    disjoint_sets.into_values()
}

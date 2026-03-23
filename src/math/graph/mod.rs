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

pub fn disjoint_set_union<T>(graph_nodes: &[T], num_nodes: usize) -> IntoValues<usize, Vec<usize>>
where
    for<'a> &'a T: IntoIterator<Item = &'a usize>,
{
    let mut node_graphs = vec![vec![]; num_nodes];
    graph_nodes.iter().enumerate().for_each(|(graph, nodes)| {
        nodes
            .into_iter()
            .for_each(|&node| node_graphs[node].push(graph))
    });
    let num_graphs = graph_nodes.len();
    let mut dsu = DisjointSetUnion::new(num_graphs);
    node_graphs
        .into_iter()
        .filter(|v| v.len() >= 2)
        .for_each(|elements| {
            let first = elements[0];
            elements[1..].iter().for_each(|&s| dsu.union(first, s))
        });
    let mut graphs: HashMap<usize, Vec<usize>> = HashMap::new();
    (0..num_graphs).for_each(|s| graphs.entry(dsu.find(s)).or_default().push(s));
    graphs.into_values()
}

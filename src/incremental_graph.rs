use log::debug;
use rand::rngs::SmallRng;
use rand::Rng;

// pub use crate::builder::*;
pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;
// use crate::utils::*;
// use std::cmp::Reverse;
// use std::collections::BinaryHeap;
// use rustc_hash::FxHashSet;

// pub struct GraphParams {
//     a: f32,
//     r: usize,
//     l: usize,
// }

// Partial orderなものをTotal orderにする
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Dist<T>(pub T);

pub trait IncrementalGraph<P: PointInterface, const R: usize, T: Rng> {
    fn new() -> Self;
    fn search(&self, point: &P, l: usize, rng: &mut T) -> Vec<(Dist<f32>, u32)>;
    fn insert(&mut self, point: P, a: f32, rng: &mut T) -> u32;
    fn delete(&mut self, index: u32, a: f32);
    fn prune(&mut self, candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R];
    fn remove(&mut self, index: u32, a: f32);
    fn remove_list(&self) -> Vec<u32>;
    fn add_edge(&mut self, node_index: &u32, new_edge_node_index: u32);
}

#[derive(Clone)]
pub struct Node<P: PointInterface, const R: usize> {
    point: P,
    edges: [u32; R],
}

pub trait Storage<P: PointInterface, const R: usize, T: Rng> {
    fn new() -> Self;

    fn get(&self, index: &u32) -> Option<Node<P, R>>;

    fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>>;

    fn delete(&mut self, index: u32);

    fn remove(&mut self, index: u32);

    fn alloc(&mut self) -> u32;

    fn random_index(&self, rng: &mut T) -> u32;
}

#[cfg(test)]
mod tests {
    use crate::point::Point;

    use super::*;
    use ic_cdk::trap;
    use itertools::Itertools;
    use rand::rngs::{SmallRng, ThreadRng};
    use rand::thread_rng;
    use std::collections::BTreeMap;
    // use itertools::Itertools;
    use rustc_hash::{FxHashMap, FxHashSet};
    // use crate::utils::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    // use rustc_hash::FxHashSet;

    const R: usize = 90;
    const L: usize = 125;

    struct TestStorage<P: PointInterface> {
        btree: BTreeMap<u32, Node<P, 90>>,
        index: u32,
    }

    impl<P, T> Storage<P, R, T> for TestStorage<P>
    where
        P: PointInterface,
        T: Rng,
    {
        fn new() -> Self {
            Self {
                btree: BTreeMap::new(),
                index: 0,
            }
        }

        fn get(&self, index: &u32) -> Option<Node<P, R>> {
            self.btree.get(index).cloned()
        }

        fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>> {
            self.btree.insert(index, node)
        }

        fn delete(&mut self, _index: u32) {
            todo!()
        }

        fn remove(&mut self, _index: u32) {
            todo!()
        }

        fn alloc(&mut self) -> u32 {
            self.index += 1;
            self.index
        }

        fn random_index(&self, rng: &mut T) -> u32 {
            let keys: Vec<_> = self.btree.keys().collect();

            let Some(&random_key) = keys.get(rng.gen_range(0..keys.len())) else {
                panic!("btree is empty")
            };

            *random_key
        }
    }

    pub struct Graph<P: PointInterface, S: Storage<P, R, T>, const R: usize, T>
    where
        T: Rng,
    {
        storage: S,
        phantom1: std::marker::PhantomData<P>,
        phantom2: std::marker::PhantomData<T>,
    }

    impl<T: PartialEq> Eq for Dist<T> {}

    impl<T: PartialOrd> Ord for Dist<T> {
        fn cmp(&self, other: &Dist<T>) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).expect("NaN was compared.")
        }
    }

    const DIM: usize = 384;

    impl<P, S, const R: usize, T> IncrementalGraph<P, R, T> for Graph<P, S, R, T>
    where
        P: PointInterface,
        S: Storage<P, R, T>,
        T: Rng,
    {
        fn new() -> Self {
            let mut storage = S::new();
            let init_node = Node {
                point: P::from_f32_vec(vec![0f32; DIM]),
                edges: [0; R],
            };
            storage.set(0, init_node);
            Self {
                storage,
                phantom1: std::marker::PhantomData,
                phantom2: std::marker::PhantomData,
            }
        }

        fn search(&self, query: &P, l: usize, rng: &mut T) -> Vec<(Dist<f32>, u32)> {
            let entry_index = self.storage.random_index(rng); // should be random later
            debug!("entry_index: {entry_index}");
            let entry_node = self.storage.get(&entry_index).unwrap();
            let dist_from_entry = Dist(query.distance(&entry_node.point));

            // let mut dist_cache = FxHashMap::default(); // Set of nodes that have been calculated the distance from query.
            let mut visited_nodes_and_dists = FxHashMap::default(); // Set of nodes that have been visited.
            let mut touched_nodes = FxHashSet::default();
            let mut heap: BinaryHeap<Reverse<(Dist<f32>, u32)>> = BinaryHeap::new();

            visited_nodes_and_dists.insert(entry_index, dist_from_entry);
            heap.push(Reverse((dist_from_entry, entry_index)));

            // Pushing uncompared edge nodes into heap
            for edge_node_index in entry_node.edges {
                // If the edge node has not been compared before,
                if !touched_nodes.insert(edge_node_index) {
                    let edge_node = self.storage.get(&edge_node_index).expect("msg");
                    let dist = Dist(query.distance(&edge_node.point));
                    heap.push(Reverse((dist, edge_node_index)));
                }
            }

            let mut num_skips = 0;

            while let Some(Reverse((dist, node_index))) = heap.pop() {
                if num_skips > l {
                    break;
                }

                debug!("visit node: {node_index}");

                // If the node already has been visited, skip this node.
                let edge_nodes = if visited_nodes_and_dists.insert(node_index, dist).is_some() {
                    num_skips += 1;
                    debug!("skip");
                    continue;
                } else {
                    debug!("non");
                    num_skips = 0;
                    self.storage.get(&node_index).expect("")
                };

                // Pushing uncompared edge nodes into heap
                for edge_node_index in edge_nodes.edges {
                    // If the edge node has not been compared before,
                    if !touched_nodes.insert(edge_node_index) {
                        let edge_node = self.storage.get(&edge_node_index).expect("msg");
                        let dist = Dist(query.distance(&edge_node.point));
                        heap.push(Reverse((dist, edge_node_index)));
                    }
                }
            }

            // Return sorted visited nodes. The first k nodes might be used for results of query search.
            visited_nodes_and_dists
                .into_iter()
                .map(|v| (v.1, v.0))
                .sorted()
                // .map(|(dist, node_index)| (dist.0, node_index))
                .collect()
        }

        fn prune(&mut self, mut candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R] {
            let mut new_edge_nodes = [0; R]; // or u32::MAX

            if candidates.len() <= R {
                for (i, (_, node_index)) in candidates.into_iter().enumerate() {
                    new_edge_nodes[i] = node_index;
                }
            } else {
                let mut edge_index = 0;

                while let Some((first, rest)) = candidates.split_first() {
                    let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
                    new_edge_nodes[edge_index] = pa;

                    candidates = rest.to_vec();

                    // if α · d(p*, p') <= d(p, p') then remove p' from v
                    candidates.retain(|&(dist_xp_pd, pd)| {
                        let pa_node = self.storage.get(&pa).expect("msg");
                        let pd_node = self.storage.get(&pd).expect("msg");
                        let dist_pa_pd = pa_node.point.distance(&pd_node.point);

                        a * dist_pa_pd > dist_xp_pd.0
                    });

                    edge_index += 1;
                    if edge_index == R {
                        break;
                    }
                }
            }

            new_edge_nodes
        }

        fn insert(&mut self, point: P, a: f32, rng: &mut T) -> u32 {
            let candidates = self.search(&point, L, rng);
            debug!("candidates: {:?}", candidates);
            let edges = <Self as IncrementalGraph<P, R, T>>::prune(self, candidates, a);

            // Add new node
            let new_node = Node { point, edges };
            let new_node_index = self.storage.alloc();
            self.storage.set(new_node_index, new_node);

            // Update graph edges
            let mut edges = edges.to_vec().into_iter().filter(|i| *i != 0).collect::<Vec<_>>();
            if edges.len() < R {
                edges.push(0);
            }
            debug!("aaa: {new_node_index}: {:?}", edges);
            for edge_node_index in edges {
                // if edge_node_index == 0 {
                //     continue;
                // }
                // let edge_node = self.storage.get(&edge_node_index).expect("msg");
                self.add_edge(&edge_node_index, new_node_index);
            }

            debug!("zero: {:?}", self.storage.get(&0).unwrap().edges);

            debug!("\n");

            new_node_index
        }

        fn delete(&mut self, _index: u32, _a: f32) {
            todo!()
        }

        fn remove(&mut self, _index: u32, _a: f32) {
            todo!()
        }

        fn remove_list(&self) -> Vec<u32> {
            todo!()
        }

        fn add_edge(&mut self, node_index: &u32, new_edge_node_index: u32) {
            // debug!("node_index: {node_index}, new_edge_node_index: {new_edge_node_index}");
            let node = self.storage.get(node_index).expect("msg");
            let mut edges = node.edges.to_vec().into_iter().filter(|i| *i != 0).collect::<Vec<_>>();
            edges.push(new_edge_node_index);

            // debug!("edges: {:?}", edges);

            let new_edges = if edges.len() > R {
                let candidates = edges
                    .into_iter()
                    .map(|candidate_node_index| {
                        let candidate_node = self.storage.get(&candidate_node_index).expect("msg");
                        (
                            Dist(node.point.distance(&candidate_node.point)),
                            candidate_node_index,
                        )
                    })
                    .collect();
                // debug!("aaaaa");
                self.prune(candidates, 2.0)
            } else {
                let mut new_edge_nodes = [0; R];
                for (i, edge_node_index) in edges.into_iter().enumerate() {
                    new_edge_nodes[i] = edge_node_index;
                }
                // debug!("new_edge_nodes: {:?}", new_edge_nodes);
                new_edge_nodes
            };

            // debug!("add_edge:{node_index}:{:?}", new_edges.iter().filter(|i| **i != 0).collect::<Vec<_>>());

            self.storage.set(
                *node_index,
                Node {
                    point: node.point,
                    edges: new_edges,
                },
            );

        }
    }

    fn normalize_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|&v| v * v).sum::<f32>().sqrt(); // ユークリッドノルムを計算
        if norm.abs() < f32::EPSILON {
            // ノルムがゼロの場合（すべての要素が0）、そのまま返す
            return vector;
        }
        vector.into_iter().map(|v| v / norm).collect()
    }

    #[test]
    fn test_get() {
        let mut rng: ThreadRng = thread_rng();
        let mut graph: Graph<Point, TestStorage<Point>, 90, ThreadRng> = Graph::new();

        env_logger::init();

        for _ in 0..10 {
            graph.insert(gen_point(&mut rng), 2.0, &mut rng);
        }
    }

    fn gen_point(rng: &mut ThreadRng) -> Point {
        let new_point: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
        let new_point = normalize_to_unit_length(new_point);
        Point(new_point)
    }
}

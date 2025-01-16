use itertools::Itertools;
use log::debug;
use rand::Rng;
use rustc_hash::FxHashSet;

pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;

// Partial orderなものをTotal orderにする
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Dist<T>(pub T);

pub trait IncrementalGraph<P: PointInterface, const R: usize, T: Rng> {
    fn new(init_point: P) -> Self;
    fn search(&self, point: &P, l: usize, rng: &mut T) -> Vec<(Dist<f32>, u32)>;
    fn insert(&mut self, point: P, a: f32, l: usize, rng: &mut T) -> u32;
    fn delete(&mut self, index: u32, a: f32);
    fn prune(&mut self, candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R];
    fn remove(&mut self, index: u32, a: f32);
    fn remove_list(&self) -> Vec<u32>;
    fn add_edge(&mut self, node_index: &u32, new_edge_node_index: u32);
}


impl<T: PartialEq> Eq for Dist<T> {}

impl<T: PartialOrd> Ord for Dist<T> {
    fn cmp(&self, other: &Dist<T>) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).expect("NaN was compared.")
    }
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

pub struct Graph<P: PointInterface, S: Storage<P, R, T>, const R: usize, T>
where
    T: Rng,
{
    storage: S,
    phantom1: std::marker::PhantomData<P>,
    phantom2: std::marker::PhantomData<T>,
}


impl<P, S, const R: usize, T> IncrementalGraph<P, R, T> for Graph<P, S, R, T>
where
    P: PointInterface,
    S: Storage<P, R, T>,
    T: Rng,
{
    fn new(init_point: P) -> Self {
        let mut storage = S::new();
        let init_node = Node {
            point: init_point,
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
        let mut list: Vec<(Dist<f32>, u32)> = vec![]; // This list has working nodes that is visited or candidate for next searching.
        let mut visited_nodes = FxHashSet::default();
        let mut touched_nodes = FxHashSet::default(); // Used for avoiding recalculation of distance already in list

        // Set entry node and its edges
        let entry_node_index = self.storage.random_index(rng);
        debug!("node {entry_node_index}: entry");
        let entry_node = self.storage.get(&entry_node_index).unwrap();
        list.push((Dist(entry_node.point.distance(query)), entry_node_index));
        touched_nodes.insert(entry_node_index);

        fn merge_lists(
            list_a: Vec<(Dist<f32>, u32)>,
            list_b: Vec<(Dist<f32>, u32)>,
            max_length: usize,
        ) -> Vec<(Dist<f32>, u32)> {
            let mut new_list = Vec::new();
            let mut list_a = list_a.into_iter();
            let mut list_b = list_b.into_iter();

            let mut smallest_in_a = list_a.next();
            let mut smallest_in_b = list_b.next();

            while new_list.len() < max_length {
                match (smallest_in_a, smallest_in_b) {
                    (None, None) => break,
                    (Some(a), None) => {
                        new_list.push(a);
                        smallest_in_a = list_a.next();
                    }
                    (None, Some(b)) => {
                        new_list.push(b);
                        smallest_in_b = list_b.next();
                    }
                    (Some(a), Some(b)) => {
                        if a < b {
                            new_list.push(a);
                            smallest_in_a = list_a.next();
                        } else {
                            new_list.push(b);
                            smallest_in_b = list_b.next();
                        }
                    }
                }
            }

            new_list
        }

        loop {
            // Find an unvisited node that has smallest dist
            let Some((_, next_visit_node)) = list
                .iter()
                .find(|(_, node_index)| visited_nodes.insert(*node_index))
            else {
                // If all nodes in the list have been visited, searching is finished
                // The results are the fist k nodes in the list
                // debug!("search: break");
                break;
            };

            // debug!("search: next node {next_visit_node}");

            // Get next node from the storage
            let working_node = self.storage.get(&next_visit_node).expect("");

            // Calculate distances between query point and the edge nodes of the current node
            let candidates: Vec<_> = working_node
                .edges
                .into_iter()
                .filter_map(|edge_node_index| {
                    // If a distance of this edge node is already calculated, skip this node
                    if !touched_nodes.insert(edge_node_index) {
                        return None;
                    }

                    let edge_node = self.storage.get(&edge_node_index).expect("msg");

                    Some((Dist(edge_node.point.distance(query)), edge_node_index))
                })
                .sorted() // candidates should be order for merging to list
                .collect();

            // Merge the two lists by order
            list = merge_lists(list.clone(), candidates, l);
        }

        list
    }

    fn prune(&mut self, mut candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R] {
        let mut new_edge_nodes = [0; R]; // or u32::MAX

        if candidates.len() <= R {
            for (i, (_, node_index)) in candidates.into_iter().enumerate() {
                new_edge_nodes[i] = node_index;
            }
        } else {
            let mut edge_index = 0;

            debug!("prune: candidates.len(): {}", candidates.len());

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

    fn insert(&mut self, point: P, a: f32, l: usize, rng: &mut T) -> u32 {
        // Search the new point and use route nodes as candidates of its edges
        let candidates = self.search(&point, l, rng);
        let edges = <Self as IncrementalGraph<P, R, T>>::prune(self, candidates, a);

        // Add new node to the storage
        let new_node = Node { point, edges };
        let new_node_index = self.storage.alloc();
        self.storage.set(new_node_index, new_node);

        // Update graph edges: add the new node to its edge node's edges
        let edges = edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        debug!("node {new_node_index}: {:?}", edges);
        for edge_node_index in edges {
            self.add_edge(&edge_node_index, new_node_index);
        }

        debug!(
            "node 0: {:?}",
            self.storage
                .get(&0)
                .unwrap()
                .edges
                .to_vec()
                .into_iter()
                .filter(|i| *i != 0)
                .collect::<Vec<_>>()
        );

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

    fn add_edge(&mut self, target_node_index: &u32, new_edge_node_index: u32) {
        debug!("add_edge: {target_node_index}, new_edge_node_index: {new_edge_node_index}");
        let target_node = self.storage.get(target_node_index).expect("msg");
        let mut edges = target_node
            .edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        edges.push(new_edge_node_index);
        debug!("add_edge: {:?}", edges);

        let new_edges = if edges.len() > R {
            let candidates = edges
                .into_iter()
                .map(|candidate_node_index| {
                    let candidate_node = self.storage.get(&candidate_node_index).expect("msg");
                    (
                        Dist(target_node.point.distance(&candidate_node.point)),
                        candidate_node_index,
                    )
                })
                .sorted()
                .collect();

            self.prune(candidates, 2.0)
        } else {
            let mut new_edge_nodes = [0; R];
            for (i, edge_node_index) in edges.into_iter().enumerate() {
                new_edge_nodes[i] = edge_node_index;
            }

            new_edge_nodes
        };

        debug!(
            "add_edge: afger: node {target_node_index}: {:?}",
            new_edges
                .to_vec()
                .into_iter()
                .sorted()
                .dedup()
                .collect::<Vec<_>>()
        );

        self.storage.set(
            *target_node_index,
            Node {
                point: target_node.point,
                edges: new_edges,
            },
        );
    }
}


#[cfg(test)]
mod tests {
    use crate::point::Point;

    use super::*;
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use std::collections::BTreeMap;
    use rustc_hash::FxHashSet;

    const R: usize = 90;
    const L: usize = 125;

    fn normalize_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|&v| v * v).sum::<f32>().sqrt(); // ユークリッドノルムを計算
        if norm.abs() < f32::EPSILON {
            // ノルムがゼロの場合（すべての要素が0）、そのまま返す
            return vector;
        }
        vector.into_iter().map(|v| v / norm).collect()
    }


    struct TestStorage<P: PointInterface> {
        btree: BTreeMap<u32, Node<P, R>>,
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

    const DIM: usize = 384;

    #[test]
    fn test_get() {
        let mut rng: SmallRng = SmallRng::from_entropy();
        let mut graph: Graph<Point, TestStorage<Point>, R, SmallRng> =
            Graph::new(gen_point(&mut rng));

        env_logger::init();

        let iter_count = 10;

        let test_points: Vec<Point> = (0..iter_count)
            .into_iter()
            .map(|_| gen_point(&mut rng))
            .collect();

        for p in test_points.iter() {
            graph.insert(p.clone(), 2.0, L, &mut rng);
        }

        let hit_counts: f32 = test_points
            .par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, p| {
                    let p = p.clone();
                    let ground_truth: Vec<(Dist<f32>, u32)> = test_points
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (Dist(p.distance(q)), (i + 1) as u32))
                        .sorted()
                        .collect();

                    let result = graph.search(&p, L, rng);

                    debug!("{:?}, {:?}", ground_truth[0], result[0]);

                    let set1: FxHashSet<_> = ground_truth[0..5]
                        .into_iter()
                        .map(|(_, index)| index)
                        .collect();
                    let set2: FxHashSet<_> =
                        result[0..5].into_iter().map(|(_, index)| index).collect();

                    let intersection_count = set1.intersection(&set2).count();

                    intersection_count as f32
                },
            )
            .sum();

        println!("{:?}", hit_counts / ((5 * iter_count) as f32));
    }

    fn gen_point(rng: &mut SmallRng) -> Point {
        let new_point: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
        let new_point = normalize_to_unit_length(new_point);
        Point(new_point)
    }
}

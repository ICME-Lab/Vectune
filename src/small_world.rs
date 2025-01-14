// use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

// use vectune::{
//     utils::{sort_list_by_dist, sort_list_by_dist_v1},
//     PointInterface,
// };

use crate::utils::{sort_list_by_dist, sort_list_by_dist_v1};
use crate::PointInterface;
// use crate::traits::PointInterface,

type NodeIndex = u32;
// type EdgeIndex = NodeIndex;
type Edges = Vec<Neighbor>;
type Transferred = Vec<Neighbor>;
type Distance = f32;
type Dropped = Vec<NodeIndex>;
type Pruned = Vec<Vec<Neighbor>>;

#[derive(Clone, Copy, Debug)]
pub struct Neighbor {
    pub index: NodeIndex,
    pub distance: Distance,
}

pub struct Node<P: PointInterface> {
    edges: Edges,
    transferred: Transferred,
    point: P,
    status: PruningStatus,
}

enum PruningStatus {
    Search,
    Prune,
    Done,
}

impl<P> Node<P>
where
    P: PointInterface,
{
    pub fn new(point: P, initial_edges: Edges) -> Self {
        Self {
            edges: initial_edges,
            transferred: Vec::new(),
            point,
            status: PruningStatus::Search,
        }
    }

    pub fn flash_transferred(&mut self) -> Transferred {
        let transferred = self.transferred.clone();
        self.transferred = Vec::new();
        transferred
    }

    pub fn receive_ownership(&mut self, new_neighbors: Vec<Neighbor>) {
        // let mut transferred = self.transferred.clone();
        self.transferred.extend(new_neighbors);
        self.transferred.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Less)
        });
        self.transferred.dedup_by(|a, b| a.index == b.index);
    }
}

pub struct Constructor<P: PointInterface> {
    nodes: Vec<Node<P>>,
    queue: VecDeque<NodeIndex>,
    medoid: NodeIndex,
    r: usize,
    l: usize,
    a: f32,
}

impl<P> Constructor<P>
where
    P: PointInterface,
{
    pub fn new(points: Vec<P>, r: usize, l: usize, a: f32) -> Self {
        // let (medoid, all_edges) = Constructor::random_edges_init(&points, r);
        let points_len = points.len();
        let (medoid, all_edges) = Constructor::nn_edges_init(&points, r);
        let nodes: Vec<Node<P>> = points
            .into_iter()
            .zip(all_edges)
            .map(|(point, edges)| Node::new(point, edges))
            .collect();
        let mut deque: VecDeque<u32> = VecDeque::with_capacity(points_len + 1);
        for i in 0..points_len as u32 {
            deque.push_back(i);
        }
        Self {
            nodes,
            queue: deque,
            medoid,
            r,
            l,
            a,
        }
    }

    pub fn export(self) -> (NodeIndex, Vec<(P, Vec<NodeIndex>)>) {
        let medoid = self.medoid;

        let nodes = self
            .nodes
            .into_iter()
            .map(|node| {
                let p = node.point;
                let edges = node.edges.into_iter().map(|n| n.index).collect();
                (p, edges)
            })
            .collect();

        (medoid, nodes)
    }

    fn find_medoid(points: &Vec<P>) -> NodeIndex {
        let points_len = points.len();

        let point_dim = points[0].to_f32_vec().len();

        let centroid = points
            .par_iter()
            .fold(
                || P::from_f32_vec(vec![0.0; point_dim]),
                |acc, x| acc.add(x),
            )
            .reduce_with(|sum1, sum2| sum1.add(&sum2))
            .unwrap()
            .div(&points_len);
        let medoid = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.distance(&centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
            .unwrap()
            .0 as u32;

        medoid
    }

    pub fn prune_all(&mut self) {
        while let Some(target) = self.queue.pop_front() {
            let mut candidates = self.nodes[target as usize].edges.clone();
            let ((_, mut visited), _, _) =
                self.greedy_search(&self.nodes[target as usize].point, 1, self.l);
            if visited[0].index == target {
                visited.truncate(1)
            }
            candidates.extend(visited);
            candidates.sort_by(|a, b| a.index.cmp(&b.index));
            candidates.dedup_by(|a, b| a.index == b.index);

            let (new_edges, _pruned, _dropped) = self.robust_prune(candidates);

            let node = &mut self.nodes[target as usize];
            node.edges = new_edges;
        }
    }

    fn nn_edges_init(points: &Vec<P>, r: usize) -> (NodeIndex, Vec<Edges>) {
        assert!(points.len() < u32::MAX as usize);

        let medoid = Constructor::find_medoid(points);

        let mut all_edges: Vec<Edges> = vec![];

        let index_and_points: Vec<(usize, &P)> = points.iter().enumerate().collect();
        for (base_index, base_point) in index_and_points.iter() {
            let mut nns: Edges = index_and_points
                .iter()
                .filter_map(|(index, p)| {
                    if index == base_index {
                        None
                    } else {
                        let index = *index as u32;
                        let distance = p.distance(base_point);

                        Some(Neighbor { index, distance })
                    }
                })
                .collect();
            sort_neighbors_by_dist(&mut nns);
            nns.truncate(r);

            all_edges.push(nns);
        }

        (medoid, all_edges)
    }

    fn _nn_edges_init(points: &Vec<P>, r: usize) -> (NodeIndex, Vec<Edges>) {
        assert!(points.len() < u32::MAX as usize);

        let medoid = Constructor::find_medoid(points);

        let mut all_edges: Vec<Edges> = vec![];

        let index_and_points: Vec<(usize, &P)> = points.iter().enumerate().collect();
        for (base_index, base_point) in index_and_points.iter() {
            let mut nns: Edges = index_and_points
                .iter()
                .filter_map(|(index, p)| {
                    if index == base_index {
                        None
                    } else {
                        let index = *index as u32;
                        let distance = p.distance(base_point);

                        Some(Neighbor { index, distance })
                    }
                })
                .collect();
            sort_neighbors_by_dist(&mut nns);
            nns.truncate(r);
            all_edges.push(nns);
        }

        (medoid, all_edges)
    }

    fn _random_edges_init(points: &Vec<P>, r: usize) -> (NodeIndex, Vec<Edges>) {
        let mut rng = thread_rng();

        if points.is_empty() {
            panic!()
        }

        assert!(points.len() < u32::MAX as usize);
        let points_len = points.len();

        let medoid = Constructor::find_medoid(points);
        let mut edges: Vec<(u32, Vec<(f32, u32)>)> = (0..points_len)
            .into_iter()
            .map(|_| {
                (
                    0,                     // n_in,
                    Vec::with_capacity(r), // n_out,
                )
            })
            .collect();
        let mut shuffle_ids: Vec<u32> = (0..points_len as u32).collect();
        shuffle_ids.shuffle(&mut rng);

        (0..points_len).into_iter().for_each(|node_i| {
            let mut rng = thread_rng();

            let mut new_ids = Vec::with_capacity(r);
            let mut shuffle_iter_count = rng.gen_range(0..(points_len * 5) as u32);
            while new_ids.len() < r {
                shuffle_iter_count += 1;
                let candidate_i = shuffle_ids[(shuffle_iter_count as usize) % points_len];

                if node_i as u32 == candidate_i || new_ids.contains(&candidate_i) {
                    continue;
                }

                let n_in_count = &mut edges[candidate_i as usize].0;

                if *n_in_count >= (r + r / 3) as u32 {
                    // println!("d3: {}, {}", candidate_i, *n_in_count);

                    continue;
                } else {
                    *n_in_count += 1;
                    new_ids.push(candidate_i);
                    shuffle_iter_count = rng.gen_range(0..points_len as u32);
                }
            }

            let new_n_out: Vec<(f32, u32)> = new_ids
                .into_par_iter()
                .map(|edge_i| {
                    let dist = points[edge_i as usize].distance(&points[node_i]);
                    (dist, edge_i)
                })
                .collect();

            edges[node_i].1 = new_n_out;
        });

        let edges = edges
            .into_iter()
            .map(|(_, neighbors)| {
                neighbors
                    .into_iter()
                    .map(|(distance, index)| Neighbor { index, distance })
                    .collect()
            })
            .collect();

        (medoid, edges)
    }

    fn robust_prune(&self, mut candidates: Vec<Neighbor>) -> (Edges, Pruned, Dropped) {
        // Prunedは、targetとの距離ではなく、刈り取った根拠にしたedgeと刈り取られたnodeとの距離を返す。つまり、targetのneigborではなく、edgeとのneighbor
        // wip: candidatesがR以下ならそのままedgeとして返す。 ただしソートはする。

        sort_neighbors_by_dist(&mut candidates);

        // if candidates.len() <= self.r {
        //     todo!();
        //     // return
        // }

        let mut new_neighbors: Vec<Neighbor> = vec![];
        let mut purned_neighbors: Vec<Vec<Neighbor>> = vec![];
        let mut dropped_neighbors: Vec<NodeIndex> = vec![];

        while let Some((first, rest)) = candidates.split_first() {
            let nearest_neighbor = *first;

            if new_neighbors.len() == self.r {
                dropped_neighbors = rest.iter().map(|n| n.index).collect(); // rest candidates are marked as dropped
                break;
            } else {
                new_neighbors.push(nearest_neighbor);
            }

            // pa is p asterisk (p*), which is nearest point to p in this loop
            // let pa_dist = nearest_neighbor.distance;
            let pa = nearest_neighbor.index;

            // if α · d(p*, p') <= d(p, p') then remove p' from v
            let mut pruned = vec![];
            candidates = rest
                .iter()
                .filter_map(|&neighbor| {
                    let dist_xp_pd = neighbor.distance;
                    let pd = neighbor.index;
                    let pa_point = &self.nodes[pa as usize].point;
                    let pd_point = &self.nodes[pd as usize].point;
                    let dist_pa_pd = pa_point.distance(pd_point);

                    if self.a * dist_pa_pd > dist_xp_pd {
                        Some(neighbor)
                    } else {
                        // Prunedは、targetとの距離ではなく、刈り取った根拠にしたedgeと刈り取られたnodeとの距離を返す。つまり、targetのneigborではなく、edgeとのneighbor
                        pruned.push(Neighbor {
                            index: pd,
                            distance: dist_pa_pd,
                        });
                        None
                    }
                })
                .collect();
            purned_neighbors.push(pruned);
        }

        (new_neighbors, purned_neighbors, dropped_neighbors)
    }

    fn search(&self, target_index: &NodeIndex) -> Vec<Neighbor> {
        let target_point = &self.nodes[*target_index as usize].point;
        let ((_, visited), _, _) = self.greedy_search(target_point, 1, self.l);
        visited
    }

    pub fn greedy_search(
        &self,
        query_point: &P,
        k: usize,
        l: usize,
    ) -> ((Vec<Neighbor>, Vec<Neighbor>), u32, Vec<NodeIndex>) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.medoid;
        let mut visited: Vec<(f32, u32)> = Vec::with_capacity(l * 2);
        let mut touched = FxHashSet::default();
        touched.reserve(l * 100);

        let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(l);
        list.push((query_point.distance(&self.nodes[s as usize].point), s, true));
        let mut working = Some(list[0]);
        visited.push((list[0].0, list[0].1));
        touched.insert(list[0].1);

        let mut read_count = 0;

        let mut collector = vec![];

        while let Some((_, nearest_i, _)) = working {
            let neighbors = &self.nodes[nearest_i as usize].edges;
            let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(neighbors.len());
            for neighbor in neighbors {
                let out_i = neighbor.index;
                if !touched.contains(&out_i) {
                    read_count += 1;
                    touched.insert(out_i);
                    nouts.push((
                        query_point.distance(&self.nodes[out_i as usize].point),
                        out_i,
                        false,
                    ));
                    collector.push(out_i);
                }
            }

            sort_list_by_dist(&mut nouts);

            let mut new_list = Vec::with_capacity(l);
            let mut new_list_idx = 0;

            let mut l_idx = 0; // Index for list
            let mut n_idx = 0; // Index for dists

            working = None;

            while new_list_idx < l {
                let mut new_min = if l_idx >= list.len() && n_idx >= nouts.len() {
                    break;
                } else if l_idx >= list.len() {
                    let new_min = nouts[n_idx];
                    n_idx += 1;
                    new_min
                } else if n_idx >= nouts.len() {
                    let new_min = list[l_idx];
                    l_idx += 1;
                    new_min
                } else {
                    let l_min = list[l_idx];
                    let n_min = nouts[n_idx];

                    if l_min.0 <= n_min.0 {
                        l_idx += 1;
                        l_min
                    } else {
                        n_idx += 1;
                        n_min
                    }
                };

                let is_not_visited = !new_min.2;

                if working.is_none() && is_not_visited {
                    new_min.2 = true; // Mark as visited
                    working = Some(new_min);
                    visited.push((new_min.0, new_min.1));
                }

                new_list.push(new_min);
                new_list_idx += 1;
            }

            list = new_list;
        }

        let mut k_anns = list
            .into_iter()
            .map(|(dist, id, _)| (dist, id))
            .collect::<Vec<(f32, u32)>>();
        k_anns.truncate(k);

        sort_list_by_dist_v1(&mut visited);

        let k_ann = k_anns
            .into_iter()
            .map(|(distance, index)| Neighbor { index, distance })
            .collect();
        let visited = visited
            .into_iter()
            .map(|(distance, index)| Neighbor { index, distance })
            .collect();

        ((k_ann, visited), read_count, collector)
    }

    pub fn greedy_search_with_optimal_stopping(
        &self,
        query_point: &P,
        k: usize,
        l: usize,
    ) -> ((Vec<Neighbor>, Vec<Neighbor>), u32) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.medoid;
        let mut visited: Vec<(f32, u32)> = Vec::with_capacity(l * 2);
        let mut touched = FxHashSet::default();
        touched.reserve(l * 100);

        let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(l);
        list.push((query_point.distance(&self.nodes[s as usize].point), s, true));
        let mut working = Some(list[0]);
        visited.push((list[0].0, list[0].1));
        touched.insert(list[0].1);

        let mut read_count = 0;

        let mut rng = thread_rng();

        while let Some((_, nearest_i, _)) = working {
            let neighbors = &self.nodes[nearest_i as usize].edges;

            let mut not_touched: Vec<_> = neighbors
                .into_iter()
                .filter(|n| !touched.contains(&n.index))
                .collect();
            not_touched.shuffle(&mut rng);

            let mut nouts = if not_touched.len() < 4 {
                let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(not_touched.len());
                for neighbor in not_touched {
                    let out_i = neighbor.index;
                    read_count += 1;
                    touched.insert(out_i);
                    nouts.push((
                        query_point.distance(&self.nodes[out_i as usize].point),
                        out_i,
                        false,
                    ))
                }
                nouts
            } else {
                // optimal stopping のkを決める n/e
                let k = (not_touched.len() as f32 / 2.718) as usize;
                // let k = (not_touched.len() as f32 / 1.5) as usize;
                if k == 0 {
                    println!("k is 0, not_touched.len(): {}", not_touched.len())
                }
                // 最初のk個は全て入れる。
                let mut nouts: Vec<(f32, u32, bool)> = not_touched[0..k]
                    .iter()
                    .map(|n| {
                        let out_i = n.index;
                        read_count += 1;
                        let out_point = &self.nodes[out_i as usize].point;
                        (query_point.distance(&out_point), out_i, false)
                    })
                    .collect();
                // k個のうち、最も評価の高いノードを候補とする。
                let candidate_score = nouts
                    .iter()
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less))
                    .unwrap()
                    .0;
                // 候補よりも良い評価もつものが現れた段階で終了する。
                for n in not_touched[k..].into_iter() {
                    let out_i = n.index;
                    // let (out_point, out_edges) = getter(&out_i);
                    read_count += 1;
                    let out_point = &self.nodes[out_i as usize].point;
                    let dist = query_point.distance(&out_point);
                    nouts.push((dist, out_i, false));
                    touched.insert(out_i);

                    if candidate_score > dist {
                        break;
                    }
                }
                nouts
            };

            // let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(neighbors.len());
            // for neighbor in neighbors {
            //     let out_i = neighbor.index;
            //     if !touched.contains(&out_i) {
            //         read_count += 1;
            //         touched.insert(out_i);
            //         nouts.push((
            //             query_point.distance(&self.nodes[out_i as usize].point),
            //             out_i,
            //             false,
            //         ))
            //     }
            // }

            sort_list_by_dist(&mut nouts);

            let mut new_list = Vec::with_capacity(l);
            let mut new_list_idx = 0;

            let mut l_idx = 0; // Index for list
            let mut n_idx = 0; // Index for dists

            working = None;

            while new_list_idx < l {
                let mut new_min = if l_idx >= list.len() && n_idx >= nouts.len() {
                    break;
                } else if l_idx >= list.len() {
                    let new_min = nouts[n_idx];
                    n_idx += 1;
                    new_min
                } else if n_idx >= nouts.len() {
                    let new_min = list[l_idx];
                    l_idx += 1;
                    new_min
                } else {
                    let l_min = list[l_idx];
                    let n_min = nouts[n_idx];

                    if l_min.0 <= n_min.0 {
                        l_idx += 1;
                        l_min
                    } else {
                        n_idx += 1;
                        n_min
                    }
                };

                let is_not_visited = !new_min.2;

                if working.is_none() && is_not_visited {
                    new_min.2 = true; // Mark as visited
                    working = Some(new_min);
                    visited.push((new_min.0, new_min.1));
                }

                new_list.push(new_min);
                new_list_idx += 1;
            }

            list = new_list;
        }

        let mut k_anns = list
            .into_iter()
            .map(|(dist, id, _)| (dist, id))
            .collect::<Vec<(f32, u32)>>();
        k_anns.truncate(k);

        sort_list_by_dist_v1(&mut visited);

        let k_ann = k_anns
            .into_iter()
            .map(|(distance, index)| Neighbor { index, distance })
            .collect();
        let visited = visited
            .into_iter()
            .map(|(distance, index)| Neighbor { index, distance })
            .collect();

        ((k_ann, visited), read_count)
    }

    pub fn set_a(&mut self, a: f32) {
        self.a = a
    }
    pub fn set_r(&mut self, r: usize) {
        self.r = r
    }

    pub fn build(&mut self) {
        // println!("{}", self.a);
        /*
        WIP:
            pruneのやつは、r以下なら即終わらせる。
         */

        while let Some(target) = self.queue.pop_front() {
            println!("queue {}", self.queue.len());

            let candidates = {
                let visited: Vec<Neighbor> = match &self.nodes[target as usize].status {
                    PruningStatus::Search => self.search(&target),
                    PruningStatus::Prune => vec![],
                    PruningStatus::Done => panic!(),
                };

                let node = &mut self.nodes[target as usize];
                let candidates: Vec<Neighbor> = visited
                    .into_iter()
                    .chain(node.edges.clone())
                    .chain(node.flash_transferred())
                    .collect();

                candidates
            };

            let (new_edges, pruned, dropped) = self.robust_prune(candidates);
            let self_index: NodeIndex = target;

            // edgeを更新
            self.nodes[self_index as usize].edges = new_edges.clone();

            // 新しいedge全てに、そのedgeを根拠に刈り取ったcandidateと自分自身の所有権を渡す。
            for (edge, mut pruned_neighbors) in new_edges.into_iter().zip(pruned) {
                pruned_neighbors.push(Neighbor {
                    index: self_index,
                    distance: edge.distance,
                });
                let edge_node = &mut self.nodes[edge.index as usize];
                edge_node.receive_ownership(pruned_neighbors);

                if let PruningStatus::Done = edge_node.status {
                    self.queue.push_back(edge.index);
                    edge_node.status = PruningStatus::Prune;
                }
            }

            // droppedを処理する
            /*
            note:
            visitedを含むpruningでdropが出たとする。それらがまだSearchタスクであれば問題ないし、pruningタスクにあっても問題ない。
            doneListに入っているものでも、visitedの最後の方から外れたもとの関係は初めからないので、それらは自身の所有者いるはず。
            問題は、一度pruningされているのにも関わらず、二度目のpurningの工程でdropが出た場合、dropしたものは再度pruningが必要。
            */
            if let PruningStatus::Prune = self.nodes[self_index as usize].status {
                for dropped_index in dropped {
                    let dropped_node = &mut self.nodes[dropped_index as usize];
                    if let PruningStatus::Done = dropped_node.status {
                        self.queue.push_back(dropped_index);
                        dropped_node.status = PruningStatus::Prune;
                    }
                }
            }

            // もしtransferredに何か入っていればTaskをspawnする。
            let self_node = &mut self.nodes[self_index as usize];
            if self_node.transferred.len() != 0 {
                // Taskをspawn
                self.queue.push_back(self_index);
                self_node.status = PruningStatus::Prune;
            } else {
                self_node.status = PruningStatus::Done;
                println!("done")
            }
        }
    }
}

fn sort_neighbors_by_dist(neighbors: &mut Vec<Neighbor>) {
    neighbors.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Less)
    })
}

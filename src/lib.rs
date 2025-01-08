/*
    Vectune is a lightweight VectorDB with Incremental Indexing, based on [FreshVamana](https://arxiv.org/pdf/2105.09613.pdf).
    Copyright © ClankPan 2024.
*/

use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::thread_rng;
use rand::Rng;
use rand::seq::SliceRandom;
// use rayon::iter::IntoParallelRefIterator;
// use rayon::iter::ParallelIterator;
// use rand::SeedableRng;
// use itertools::Itertools;
// use rand::rngs::SmallRng;
// use rand::seq::SliceRandom;
// use rayon::iter::IntoParallelIterator;
// use rayon::iter::ParallelIterator;
use rustc_hash::FxHashSet;
use small_world::Constructor;
// use std::collections::BinaryHeap;
// use std::sync::atomic::{AtomicBool, Ordering};

pub mod builder;
pub mod traits;
pub mod utils;
pub mod small_world;

#[cfg(test)]
mod tests;

pub use crate::builder::*;
pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;
// pub use crate::traits::PointHelper;
use crate::utils::*;

// pub fn insert() {

// }

// pub fn delete() {

// }

// pub fn suspend() {

// }

/// Performs Greedy-Best-First-Search on a Graph that implements the GraphInterface trait.
///
/// Returns a tuple containing the list of k search results and the list of explored nodes.
///
/// Removes the nodes returned by graph.cemetery() from the results.
///
/// # Examples
///
/// ```ignore
/// let (results, visited) = vectune::search(&mut graph, &Point(query), 50);
/// ```
///
pub fn search<P, G>(graph: &mut G, query_point: &P, k: usize) -> (Vec<(f32, u32)>, Vec<(f32, u32)>)
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    // k-anns, visited
    let builder_l = graph.size_l();
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(builder_l);
    let s = graph.start_id();
    let (s_point, _) = graph.get(&s);
    // list.push((query_point.distance(&s_point), s, true));
    list.push((query_point.distance(&s_point), s, true));
    let mut working = Some(list[0]);
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    // let mut rng = SmallRng::seed_from_u64(1234456778);

    while let Some((_, nearest_i, _)) = working {
        let (_, nearest_n_out) = graph.get(&nearest_i);
        let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(nearest_n_out.len());
        for out_i in nearest_n_out {
            if !touched.contains(&out_i) {
                touched.insert(out_i);
                let (out_point, _) = graph.get(&out_i);
                nouts.push((query_point.distance(&out_point), out_i, false))
            }
        }
        sort_list_by_dist(&mut nouts);


        // let mut not_touched: Vec<_> = nearest_n_out.into_iter().filter(|out_i| !touched.contains(&out_i)).collect();
        // not_touched.shuffle(&mut rng);
        // // not_touched.iter().for_each(|i| { touched.insert(*i); });

        // let mut nouts = if not_touched.len() < 5 {
        //     let mut nouts: Vec<(f32, u32, bool)> = vec![];
        //     for out_i in not_touched{
        //         let (out_point, out_edges) = graph.get(&out_i);
        //         let dist = query_point.distance(&out_point);
        //         nouts.push((dist, out_i, false));
        //         touched.insert(out_i);
        //     }
        //     nouts
        // } else {
        //     // optimal stopping のkを決める n/e
        //     let k = (not_touched.len() as f32 / 2.718) as usize;
        //     // let k = (not_touched.len() as f32 / 1.5) as usize;
        //     if k == 0 {
        //         println!("k is 0, not_touched.len(): {}", not_touched.len())
        //     }
        //     // 最初のk個は全て入れる。
        //     let mut nouts: Vec<(f32, u32, bool)> = not_touched[0..k].iter().map(|out_i| {
        //         let (out_point, _out_edges) = graph.get(out_i);
        //         (query_point.distance(&out_point), *out_i, false)
        //     }).collect();
        //     // k個のうち、最も評価の高いノードを候補とする。
        //     let candidate_score = nouts.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less)).unwrap().0;
        //     // 候補よりも良い評価もつものが現れた段階で終了する。
        //     let mut count = 0;
        //     for out_i in not_touched[k..].into_iter() {
        //         let (out_point, _out_edges) = graph.get(&out_i);
        //         let dist = query_point.distance(&out_point);
        //         nouts.push((dist, *out_i, false));
        //         touched.insert(*out_i);

        //         count += 1;
                
        //         if candidate_score > dist {
        //             break
        //         }
        //     }
        //     // for _ in k+count..not_touched.len() {
        //     //     let _ = getter(&0);
        //     // }
        //     nouts
        // };
        // sort_list_by_dist(&mut nouts);




        let mut new_list = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        let mut l_idx = 0; // Index for list
        let mut n_idx = 0; // Index for dists

        working = None;

        while new_list_idx < builder_l {
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

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !graph.cemetery().contains(&new_min.1) || is_not_visited {
                new_list.push(new_min);
                new_list_idx += 1;
            }
        }

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    (k_anns, visited)
}



pub fn search_with_analysis_v2<P>(
    ann: &Vec<(P, Vec<u32>)>,
    query_point: &P,
    k: usize,
    l: usize,
    s: u32,
    cemetery: Vec<u32>
) -> ((Vec<(f32, u32)>, Vec<(f32, u32)>), (usize, usize))
where
    P: PointInterface,
{
    // for analysis
    let mut get_count = 0;
    let mut waste_cout = 0;

    // k-anns, visited
    let builder_l = l;
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
    let (s_point, s_edges) = ann[s as usize].clone();

    let mut getter = |id: &u32| -> (P, Vec<u32>) {
        get_count += 1;
        ann[*id as usize].clone()
    };

    list.push((query_point.distance(&s_point), s, true, s_edges));
    let mut working = Some(list[0].clone());
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    while let Some((_dist_of_working_and_query, _working_node_i, _is_visited, working_node_out)) =
        working
    {

        // println!("working_node_i: {}", working_node_i);


        let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = Vec::new();
        for out_i in working_node_out {
            if !touched.contains(&out_i) {
                // touched.insert(out_i); // ここでtouchしてしまうと次に触られなくなる。
                let (out_point, out_edges) = getter(&out_i);
                nouts.push((query_point.distance(&out_point), out_i, false, out_edges))
            }
        }
        sort_list_by_dist_v3(&mut nouts);
        nouts.truncate(3);
        nouts.iter().for_each(|(_, i, _, _)| { touched.insert(*i); });


        let mut new_list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        working = None;

        let mut list_iter = list.into_iter().peekable();
        let mut nouts_iter = nouts.into_iter().peekable();

        while new_list_idx < builder_l {
            let mut new_min = (match (list_iter.peek(), nouts_iter.peek()) {
                (None, None) => break,
                (Some(_), None) => list_iter.next(),
                (None, Some(_)) => nouts_iter.next(),
                (Some((l_min_dist, _, _, _)), Some((n_min_dist, _, _, _))) => {
                    if l_min_dist <= n_min_dist {
                        list_iter.next()
                    } else {
                        nouts_iter.next()
                    }
                }
            })
            .unwrap();

            let is_not_visited = !new_min.2;

            // Finding Your Next Visit
            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                visited.push((new_min.0, new_min.1));
                working = Some(new_min.clone());
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !cemetery.contains(&new_min.1) || is_not_visited {
                // Remove duplicate nodes
                if new_list.last().map_or(true, |&(_, last_item_index, _, _)| {
                    let is_same = last_item_index == new_min.1;
                    if is_same {
                        waste_cout += 1;
                    }
                    !is_same
                }) {
                    // Add this node to list
                    new_list.push(new_min);
                    new_list_idx += 1;
                }
            }
        }

        waste_cout += nouts_iter.count();

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    ((k_anns, visited), (get_count, waste_cout))
}

pub fn search_with_kann<P, G>(graph: &mut G, query_point: &P, k: usize) -> ((Vec<(f32, u32)>, Vec<(f32, u32)>),  (usize, usize))
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    // k-anns, visited
    let builder_l = graph.size_l();
    assert!(builder_l >= k);

    let mut get_count = 0;
    let mut waste_cout = 0;

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(builder_l);
    let s = graph.start_id();
    let (s_point, _) = graph.get(&s);
    // list.push((query_point.distance(&s_point), s, true));
    list.push((query_point.distance(&s_point), s, true));
    let mut working = Some(list[0]);
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    // let mut rng = SmallRng::seed_from_u64(1234456778);

    while let Some((_, nearest_i, _)) = working {
        let (base_point, nearest_n_out) = graph.get(&nearest_i);

        /* --- kann --- */
        let size_l_build = 5;
        let a = 1.1;
        let r = 15;
        let size_l_search = 10;
        // let k = 5;

        let edge_points: Vec<_> = nearest_n_out.iter().map(|out_i| graph.get(out_i).0).collect();
        let constructor = Constructor::new(edge_points, r, size_l_build, a);
        let ((_kann, _visited), _read_count, read_nodes) =
                    // constructor.greedy_search_with_optimal_stopping(&base_point, k, size_l_search);
                    constructor.greedy_search(&base_point, k, size_l_search);
        // println!("read_nodes: {}", read_nodes.len());
        let nearest_n_out: Vec<_> = read_nodes.into_iter().map(|edge_local_index| nearest_n_out[edge_local_index as usize]).collect();
        
        
        /* ------------ */
        let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(nearest_n_out.len());
        for out_i in nearest_n_out {
            if !touched.contains(&out_i) {
                touched.insert(out_i);
                let (out_point, _) = graph.get(&out_i);
                nouts.push((query_point.distance(&out_point), out_i, false));
                get_count += 1;
            }
        }
        sort_list_by_dist(&mut nouts);

        let mut new_list: Vec<(f32, u32, bool)> = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        working = None;

        let mut list_iter = list.into_iter().peekable();
        let mut nouts_iter = nouts.into_iter().peekable();

        while new_list_idx < builder_l {
            let mut new_min = (match (list_iter.peek(), nouts_iter.peek()) {
                (None, None) => break,
                (Some(_), None) => list_iter.next(),
                (None, Some(_)) => nouts_iter.next(),
                (Some((l_min_dist, _, _)), Some((n_min_dist, _, _))) => {
                    if l_min_dist <= n_min_dist {
                        list_iter.next()
                    } else {
                        nouts_iter.next()
                    }
                }
            })
            .unwrap();

            let is_not_visited = !new_min.2;

            // Finding Your Next Visit
            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                visited.push((new_min.0, new_min.1));
                working = Some(new_min);
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !graph.cemetery().contains(&new_min.1) || is_not_visited {
                // Remove duplicate nodes
                if new_list.last().map_or(true, |&(_, last_item_index, _)| {
                    let is_same = last_item_index == new_min.1;
                    if is_same {
                        waste_cout += 1;
                    }
                    !is_same
                }) {
                    // Add this node to list
                    new_list.push(new_min);
                    new_list_idx += 1;
                }
            }
        }

        waste_cout += nouts_iter.count();

        list = new_list;

    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    ((k_anns, visited),  (get_count, waste_cout))
}

pub fn search_with_optimal_stopping<P, G>(
    graph: &mut G, query_point: &P, k: usize, rng: &mut SmallRng
) -> ((Vec<(f32, u32)>, Vec<(f32, u32)>), (usize, usize))
where
    P: PointInterface,
    G: GraphInterface<P>
{


    // let mut _rng = thread_rng();
    // let rng = &mut _rng;

    // for analysis
    let mut get_count = 0;
    let mut waste_cout = 0;

    // k-anns, visited
    let builder_l = graph.size_l();
    assert!(builder_l >= k);
    let cemetery = graph.cemetery();

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
    let s = graph.start_id();
    let (s_point, s_edges) = graph.get(&s);
    // let (s_point, s_edges) = ann[s as usize].clone();

    let mut getter = |id: &u32| -> (P, Vec<u32>) {
        get_count += 1;
        // ann[*id as usize].clone()
        graph.get(id)
    };

    list.push((query_point.distance(&s_point), s, true, s_edges));
    let mut working = Some(list[0].clone().3);
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    while let Some(working_node_out) =
        working
    {

        let mut not_touched: Vec<_> = working_node_out.into_iter().filter(|out_i| !touched.contains(&out_i)).collect();
        not_touched.shuffle(rng);
        // not_touched.iter().for_each(|i| { touched.insert(*i); });

        let mut nouts = if not_touched.len() < 5 {
            // println!("debug");
            let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = vec![];
            for out_i in not_touched{
                let (out_point, out_edges) = getter(&out_i);
                let dist = query_point.distance(&out_point);
                nouts.push((dist, out_i, false, out_edges));
                touched.insert(out_i);
            }
            nouts
        }
        
        // else if not_touched.len() < 100 {
        //     //optimal stopping のkを決める n/e
        //     // let k = (not_touched.len() as f32 * 0.37) as usize;
        //     // let k = (not_touched.len() as f32 * 0.2718) as usize;
        //     let k = (not_touched.len() as f32 * 0.2) as usize;
        //     // let k = (not_touched.len() as f32 / 5.0) as usize;
        //     if k == 0 {
        //         println!("k is 0, not_touched.len(): {}", not_touched.len())
        //     }
        //     // 最初のk個は全て入れる。
        //     let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = not_touched[0..k].iter().map(|out_i| {
        //         let (out_point, out_edges) = getter(out_i);
        //         touched.insert(*out_i);
        //         (query_point.distance(&out_point), *out_i, false, out_edges)
        //     }).collect();
        //     // k個のうち、最も評価の高いノードを候補とする。
        //     let candidate_score = nouts.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less)).unwrap().0;
        //     // 候補よりも良い評価もつものが現れた段階で終了する。
        //     let mut not_not_touched = not_touched[k..].into_iter();
        //     while let Some(out_i) = not_not_touched.next() {
        //         let (out_point, out_edges) = getter(&out_i);
        //         let dist = query_point.distance(&out_point);
        //         nouts.push((dist, *out_i, false, out_edges));
        //         touched.insert(*out_i);
                
        //         if candidate_score > dist {
        //             break
        //         }
        //     }
        //     // not_not_touched.for_each(|out_i| {touched.insert(*out_i);});

        //     nouts
        // }
        
        else {
            let not_touched_len = not_touched.len();
            let table = [0.258, 0.448, 0.564, 0.641, 0.695, 0.735, 0.766, 0.790, 0.810, 0.827, 1.0];
            // let table = [0.25, 0.564, 0.735, 0.766, 0.790, 0.810, 0.827, 1.0];
            let thr: Vec<usize> = table.into_iter().map(|t| (not_touched_len as f32 * t) as usize).collect();
            // optimal stopping のkを決める n/e
            let k = thr[0];
            // let k = (not_touched.len() as f32 * 0.2718) as usize;
            // let k = (not_touched.len() as f32 / 5.0) as usize;
            if k == 0 {
                println!("k is 0, not_touched.len(): {}", not_touched.len())
            }
            // 最初のk個は全て入れる。
            let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = not_touched[0..k].iter().map(|out_i| {
                let (out_point, out_edges) = getter(out_i);
                touched.insert(*out_i);
                (query_point.distance(&out_point), *out_i, false, out_edges)
            }).collect();
            // k個のうち、最も評価の高いノードを候補とする。
            let mut candidate_scores: Vec<f32> = nouts.iter().map(|(dist, _, _, _)| *dist).sorted_by(|a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Less)).collect();
            
            // 候補よりも良い評価もつものが現れた段階で終了する。
            let mut relative_rank = 0;
            for (pos, out_i) in not_touched[k..].into_iter().enumerate().map(|(u, i)| (u+k, i)) {
                let (out_point, out_edges) = getter(&out_i);
                let dist = query_point.distance(&out_point);
                nouts.push((dist, *out_i, false, out_edges));
                touched.insert(*out_i);
                
                if candidate_scores[relative_rank] > dist {
                    break
                } else {
                    candidate_scores.push(dist);
                    candidate_scores.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Less));
                }

                while pos > thr[relative_rank] {
                    relative_rank += 1;
                }

            }

            nouts









            // // optimal stopping のkを決める n/e
            // let k = (not_touched.len() as f32 * 0.37) as usize;
            // // let k = (not_touched.len() as f32 * 0.2718) as usize;
            // // let k = (not_touched.len() as f32 / 5.0) as usize;
            // if k == 0 {
            //     println!("k is 0, not_touched.len(): {}", not_touched.len())
            // }
            // // 最初のk個は全て入れる。
            // let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = not_touched[0..k].iter().map(|out_i| {
            //     let (out_point, out_edges) = getter(out_i);
            //     touched.insert(*out_i);
            //     (query_point.distance(&out_point), *out_i, false, out_edges)
            // }).collect();
            // // k個のうち、最も評価の高いノードを候補とする。
            // let candidate_score = nouts.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less)).unwrap().0;
            // // 候補よりも良い評価もつものが現れた段階で終了する。
            // let mut not_not_touched = not_touched[k..].into_iter();
            // while let Some(out_i) = not_not_touched.next() {
            //     let (out_point, out_edges) = getter(&out_i);
            //     let dist = query_point.distance(&out_point);
            //     nouts.push((dist, *out_i, false, out_edges));
            //     touched.insert(*out_i);
                
            //     if candidate_score > dist {
            //         break
            //     }
            // }
            // not_not_touched.for_each(|out_i| {touched.insert(*out_i);});

            // nouts
        };

        sort_list_by_dist_v3(&mut nouts);


        let mut new_list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        working = None;

        let mut list_iter = list.into_iter().peekable();
        let mut nouts_iter = nouts.into_iter().peekable();

        while new_list_idx < builder_l {
            let mut new_min = (match (list_iter.peek(), nouts_iter.peek()) {
                (None, None) => break,
                (Some(_), None) => list_iter.next(),
                (None, Some(_)) => nouts_iter.next(),
                (Some((l_min_dist, _, _, _)), Some((n_min_dist, _, _, _))) => {
                    if l_min_dist <= n_min_dist {
                        list_iter.next()
                    } else {
                        nouts_iter.next()
                    }
                }
            })
            .unwrap();

            let is_not_visited = !new_min.2;

            // Finding Your Next Visit
            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                visited.push((new_min.0, new_min.1));
                working = Some(new_min.3.clone());
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !cemetery.contains(&new_min.1) || is_not_visited {
                // Remove duplicate nodes
                if new_list.last().map_or(true, |&(_, last_item_index, _, _)| {
                    let is_same = last_item_index == new_min.1;
                    if is_same {
                        waste_cout += 1;
                    }
                    !is_same
                }) {
                    // Add this node to list
                    new_list.push(new_min);
                    new_list_idx += 1;
                }
            }
        }

        waste_cout += nouts_iter.count();

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    ((k_anns, visited), (get_count, waste_cout))
}


pub enum Tree<Point> {
    Node(u8, Box<Tree<Point>>, Box<Tree<Point>>),
    Leaf(Vec<(Point, u8)>),
}

pub struct TreeUtils<Point>
where
    Point: PointInterface,
{
    random_splitter_points: Vec<(Point, Point)>,
    max_depth: usize,
}

impl<Point> TreeUtils<Point>
where
    Point: PointInterface,

{
    pub fn new(random_splitter_points: Vec<(Point, Point)>, max_depth: usize) -> Self {
        Self {
            random_splitter_points,
            max_depth,
        }
    }

    pub fn build(&self, points: Vec<Point>) -> Tree<Point> {
        self.rec_build(points.iter().enumerate().map(|(i, p)| (p, i as u8)).collect(), self.max_depth)
    }


    fn rec_build(&self, points: Vec<(&Point, u8)>, depth: usize) -> Tree<Point> {

        // if points.len() <= self.max_leaf_num_vectors {
        if depth == 1 {
            Tree::Leaf(points.into_iter().map(|(p, i)| (p.clone(), i)).collect())
        } else {

            let (splitter, left, right): (u8, Vec<(&Point, u8)>, Vec<(&Point, u8)>) = self.random_splitter_points.iter().enumerate().map(|(splitter_index, splitter)| {
                let (left, right) = self.random_hyperplane_split(points.clone(), splitter);
                let abs_diff = left.len().abs_diff(right.len());

                (abs_diff, (splitter_index as u8, left, right))
            }).min_by(|(diff_a, _), (diff_b, _)| diff_a.cmp(diff_b)).unwrap().1;
            let left_node = self.rec_build(left, depth-1);
            let right_node = self.rec_build(right, depth-1);

            Tree::Node(splitter.clone(), Box::new(left_node), Box::new(right_node))
        }                    
    }

    fn random_hyperplane_split<'a>(&self, points: Vec<(&'a Point, u8)>, splitter: &(Point, Point)) -> (Vec<(&'a Point, u8)>, Vec<(&'a Point, u8)>) {
        let mut left = vec![];
        let mut right = vec![];
        points.into_iter().for_each(|p| {
            let dist_l = p.0.distance(&splitter.0);
            let dist_r = p.0.distance(&splitter.1);
            if dist_l <= dist_r {
                left.push(p)
            } else {
                right.push(p)
            }
        });

        (left, right)
    }

    pub fn search_tree(&self, tree: Tree<Point>, query: &Point) -> Vec<(Point, u8)> {
        match tree {
            Tree::Node(splitter_index, left, right) => {
                let splitter = &self.random_splitter_points[splitter_index as usize];
                let dist_l = query.distance(&splitter.0);
                let dist_r = query.distance(&splitter.1);

                if dist_l <= dist_r {
                    self.search_tree(*left, query)
                } else {
                    self.search_tree(*right, query)
                }
            },
            Tree::Leaf(points) => points
        }
      }
}

pub fn search_with_analysis<P, G>(
    graph: &mut G,
    query_point: &P,
    k: usize,
    l: usize,
) -> ((Vec<(f32, u32)>, Vec<(f32, u32)>), (usize, usize))
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    // for analysis
    let mut get_count = 0;
    let mut waste_cout = 0;

    // k-anns, visited
    let builder_l = l;
    let s = graph.start_id();
    let cemetery = graph.cemetery();
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
    let (s_point, s_edges) = graph.get(&s);

    let mut getter = |id: &u32| -> (P, Vec<u32>) {
        get_count += 1;
        graph.get(id)
    };

    list.push((query_point.distance(&s_point), s, true, s_edges));
    let mut working = Some(list[0].clone());
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    let mut rng = thread_rng();

    // let mut loop_count = 0;

    while let Some((_dist_of_working_and_query, working_node_i, _is_visited, working_node_out)) =
        working
    {

        // println!("working_node_i: {}", working_node_i);


        let (working_point, _) = getter(&working_node_i);


        let mut nout_candidates: Vec<(f32, u32, bool, Vec<u32>)> = Vec::new();
        let _working_node_nearest = working_node_out.into_iter().map(|out_i| {
            let (out_point, out_edges) = getter(&out_i);

            // if !touched.contains(&out_i) {
            //     nout_candidates.push((query_point.distance(&out_point), out_i, false, out_edges));
            // }

            nout_candidates.push((query_point.distance(&out_point), out_i, false, out_edges));

            (working_point.distance(&out_point), out_i)
        }).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less)).unwrap();

        

        sort_list_by_dist_v3(&mut nout_candidates);

        // let mut nouts = nout_candidates;
        // nouts.truncate(5);
        // nouts.remove(rng.gen_range(0..nouts.len()));
        // nouts.remove(rng.gen_range(0..nouts.len()));

        let cut_len = 1;
        let nouts = if nout_candidates.len() >= cut_len {
            let mut nouts = nout_candidates[0..cut_len].to_vec();

            if rng.gen_range(0..4) == 0 {
                nouts.remove(0);
            }
    
            let mut rest = nout_candidates[cut_len..].to_vec();
            while rest.len() > 15 {
                rest.remove(rng.gen_range(0..rest.len()));
            }
            nouts.extend(rest);


            // if nouts.iter().find(|(_, out_i, _, _)| {
            //     *out_i == working_node_nearest.1
            // }).is_none() {
            //     let (out_point, out_edges) = getter(&working_node_nearest.1);
            //     nouts.push((query_point.distance(&out_point), working_node_nearest.1, false, out_edges));
            //     sort_list_by_dist_v3(&mut nouts);
            // }

            nouts

        } else {
            nout_candidates
        };


        // 一度タッチしたやつをフィルターする。
        let nouts: Vec<(f32, u32, bool, Vec<u32>)> = nouts.into_iter().filter(|(_, out_i, _, _)| !touched.contains(&out_i)).collect();
        // それ以外をタッチする
        nouts.iter().for_each(|(_, i, _, _)| { touched.insert(*i); });


        let mut new_list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        working = None;

        let mut list_iter = list.into_iter().peekable();
        let mut nouts_iter = nouts.into_iter().peekable();

        while new_list_idx < builder_l {
            let mut new_min = (match (list_iter.peek(), nouts_iter.peek()) {
                (None, None) => break,
                (Some(_), None) => list_iter.next(),
                (None, Some(_)) => nouts_iter.next(),
                (Some((l_min_dist, _, _, _)), Some((n_min_dist, _, _, _))) => {
                    if l_min_dist <= n_min_dist {
                        list_iter.next()
                    } else {
                        nouts_iter.next()
                    }
                }
            })
            .unwrap();

            let is_not_visited = !new_min.2;

            // Finding Your Next Visit
            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                visited.push((new_min.0, new_min.1));
                working = Some(new_min.clone());
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !cemetery.contains(&new_min.1) || is_not_visited {
                // Remove duplicate nodes
                if new_list.last().map_or(true, |&(_, last_item_index, _, _)| {
                    let is_same = last_item_index == new_min.1;
                    if is_same {
                        waste_cout += 1;
                    }
                    !is_same
                }) {
                    // Add this node to list
                    new_list.push(new_min);
                    new_list_idx += 1;
                }
            }
        }

        waste_cout += nouts_iter.count();

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    ((k_anns, visited), (get_count, waste_cout))
}


pub fn _search_with_analysis<P, G>(
    graph: &mut G,
    query_point: &P,
    k: usize,

    _pq: &Vec<[u8; 4]>,
    pq_point_table: &[[P; 256]; 4],

    query_pq: &[u8; 4],

    _pq_num_divs: &usize,
) -> ((Vec<(f32, u32)>, Vec<(f32, u32)>), (usize, usize))
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    // for analysis
    let mut get_count = 0;
    let mut waste_cout = 0;

    // k-anns, visited
    let builder_l = graph.size_l();
    let s = graph.start_id();
    let cemetery = graph.cemetery();
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
    let (s_point, s_edges) = graph.get(&s);

    let mut getter = |id: &u32| -> (P, Vec<u32>) {
        // get_count += 1;
        graph.get(id)
    };

    list.push((query_point.distance(&s_point), s, true, s_edges));
    let mut working = Some(list[0].clone());
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    let _query_pq: Vec<&P> = query_pq
        .into_iter()
        .zip(pq_point_table.iter())
        .map(|(index, table)| &table[*index as usize])
        .collect();

    while let Some((_dist_of_working_and_query, working_node_i, _is_visited, working_node_out)) =
        working
    {

        let (working_node_point, _) = getter(&working_node_i); // wip todo: workingのSomeに入れる。

        // より近い最初の1/3しか使わないようにする。
        let mut nouts_candidates: Vec<(f32, u32, P, Vec<u32>)> = working_node_out
            .into_iter()
            .filter_map(|out_i| {
                if !touched.contains(&out_i) {
                    touched.insert(out_i);

                    // get_count += 1;

                    let (out_point, out_edges) = getter(&out_i);

                    let dist_of_working_node_and_its_edge = out_point.distance(&working_node_point);

                    Some((dist_of_working_node_and_its_edge, out_i, out_point, out_edges))
                } else {
                    None
                }

            })
            .collect();
        nouts_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
        let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = Vec::new();
        let mut captured = false;

        let nouts_candidates = nouts_candidates.into_iter();
        for (dist_of_working_node_and_its_edge, out_i, out_point, out_edges) in nouts_candidates {

            let dist_of_query_and_out_node = query_point.distance(&out_point);
            nouts.push((dist_of_query_and_out_node, out_i, false, out_edges));

            get_count += 1;

            if captured {
                break
            }

            if dist_of_working_node_and_its_edge / 1.0 >= dist_of_query_and_out_node {
                captured = true
            }

        }

        // println!("nouts len: {}", nouts.len());

        // nouts_candidates.truncate(nouts_candidates.len() / pq_num_divs);

        // let mut nouts: Vec<(f32, u32, bool, Vec<u32>)> = nouts_candidates
        //     .into_iter()
        //     .map(|(_, out_i)| {
        //         let (out_point, out_edges) = getter(&out_i);
        //         (query_point.distance(&out_point), out_i, false, out_edges)
        //     })
        //     .collect();


        sort_list_by_dist_v3(&mut nouts);

        let mut new_list: Vec<(f32, u32, bool, Vec<u32>)> = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        working = None;

        let mut list_iter = list.into_iter().peekable();
        let mut nouts_iter = nouts.into_iter().peekable();

        while new_list_idx < builder_l {
            let mut new_min = (match (list_iter.peek(), nouts_iter.peek()) {
                (None, None) => break,
                (Some(_), None) => list_iter.next(),
                (None, Some(_)) => nouts_iter.next(),
                (Some((l_min_dist, _, _, _)), Some((n_min_dist, _, _, _))) => {
                    if l_min_dist <= n_min_dist {
                        list_iter.next()
                    } else {
                        nouts_iter.next()
                    }
                }
            })
            .unwrap();

            let is_not_visited = !new_min.2;

            // Finding Your Next Visit
            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                visited.push((new_min.0, new_min.1));
                working = Some(new_min.clone());
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !cemetery.contains(&new_min.1) || is_not_visited {
                // Remove duplicate nodes
                if new_list.last().map_or(true, |&(_, last_item_index, _, _)| {
                    let is_same = last_item_index == new_min.1;
                    if is_same {
                        waste_cout += 1;
                    }
                    !is_same
                }) {
                    // Add this node to list
                    new_list.push(new_min);
                    new_list_idx += 1;
                }
            }
        }

        waste_cout += nouts_iter.count();

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _, _)| (dist, id))
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    ((k_anns, visited), (get_count, waste_cout))
}

pub enum InsertType<P>
where
    P: PointInterface,
{
    Id(u32),
    Point(P)
}

/// Insert a new node into a Graph that implements the GraphInterface trait.
///
/// Internally, use graph.alloc() to allocate space in storage or memory and reconnect the edges.
pub fn insert<P, G>(graph: &mut G, data: InsertType<P>) -> u32
where
    P: PointInterface,
    G: GraphInterface<P>,
{

    // ic_cdk::println!("in vectne 1. :{}", ic_cdk::api::instruction_counter());

    let (new_id, new_p) = match data {
        InsertType::Id(new_id) => {
            let (new_p, _) = graph.get(&new_id);
            (new_id, new_p)
        },
        InsertType::Point(new_p) => {
            let new_id = graph.alloc(new_p.clone());
            (new_id, new_p)
        },
    };
    let r = graph.size_r();
    let a = graph.size_a();

    // ic_cdk::println!("in vectne 2. :{}", ic_cdk::api::instruction_counter());

    // [L, V] ← GreedySearch(𝑠, 𝑝, 1, 𝐿)
    let (_list, mut visited) = search(graph, &new_p, 1);

    // ic_cdk::println!("in vectne 3. :{}", ic_cdk::api::instruction_counter());

    // 𝑁out(𝑝) ← RobustPrune(𝑝, V, 𝛼, 𝑅) (Algorithm 3)
    let n_out = prune(|id| graph.get(id), &mut visited, &r, &a);

    // ic_cdk::println!("in vectne 4. :{}", ic_cdk::api::instruction_counter());

    graph.overwirte_out_edges(&new_id, n_out.clone());

    // foreach 𝑗 ∈ 𝑁out(𝑝) do
    for j in &n_out {
        // |𝑁out(𝑗) ∪ {𝑝}|
        let (j_point, mut j_n_out) = graph.get(j);
        j_n_out.push(new_id);
        j_n_out.sort();
        j_n_out.dedup();
        // if |𝑁out(𝑗) ∪ {𝑝}| > 𝑅 then
        if j_n_out.len() > r {
            // 𝑁out(𝑗) ← RobustPrune(𝑗, 𝑁out(𝑗) ∪ {𝑝}, 𝛼, 𝑅)
            let mut j_n_out_with_dist = j_n_out
                .iter()
                .map(|j_out_idx| (j_point.distance(&new_p), *j_out_idx))
                .collect::<Vec<(f32, u32)>>();
            sort_list_by_dist_v1(&mut j_n_out_with_dist);
            j_n_out = prune(|id| graph.get(id), &mut j_n_out_with_dist, &r, &a);
        }
        graph.overwirte_out_edges(j, j_n_out);
    }

    // ic_cdk::println!("in vectne 5. :{}", ic_cdk::api::instruction_counter());

    new_id
}

/// Completely removes the nodes returned by graph.cemetery() from a Graph that implements the GraphInterface trait.
pub fn delete<P, G>(graph: &mut G)
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    /* 𝑝 ∈ 𝑃 \ 𝐿𝐷 s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅ */

    // Note: 𝐿𝐷 is Deleted List
    let mut ps = Vec::new();

    // s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅
    let mut cemetery = graph.cemetery();
    cemetery.sort();
    cemetery.dedup();

    for grave_i in &cemetery {
        ps.extend(graph.backlink(grave_i))
    }
    ps.sort();
    ps.dedup();

    // 𝑝 ∈ 𝑃 \ 𝐿𝐷
    ps = diff_ids(&ps, &cemetery);

    for p in ps {
        // D ← 𝑁out(𝑝) ∩ 𝐿𝐷
        let (_, p_n_out) = graph.get(&p);
        let d = intersect_ids(&p_n_out, &cemetery);
        // C ← 𝑁out(𝑝) \ D //initialize candidate list
        let mut c = diff_ids(&p_n_out, &d);

        // foreach 𝑣 ∈ D do
        for u in &d {
            // C ← C ∪ 𝑁out(𝑣)
            // c = union_ids(&c, &self.nodes[*u].n_out);
            let (_, u_n_out) = graph.get(u);
            c.extend(u_n_out);
            c.sort();
            c.dedup();
        }

        // C ← C \ D
        /*
        Note:
            Since D's Nout may contain LD, Why pull the D instead of the LD?
            I implemented it as shown and it hit data that should have been erased, so I'll fix it to pull LD.
        */
        c = diff_ids(&c, &cemetery);

        // 𝑁out(𝑝) ← RobustPrune(𝑝, C, 𝛼, 𝑅)
        //   let (p_point, _) = self.nodes[p].p.clone();
        let (p_point, _) = graph.get(&p);
        let mut c_with_dist: Vec<(f32, u32)> = c
            .into_iter()
            .map(|id| (p_point.distance(&graph.get(&id).0), id))
            .collect();

        sort_list_by_dist_v1(&mut c_with_dist);

        /*
        Note:
            Before call robust_prune, clean Nout(p) because robust_prune takes union v and Nout(p) inside.
            It may ontain deleted points.
            The original paper does not explicitly state in Algorithm 4.
        */
        let r = graph.size_r();
        let a = graph.size_a();
        let new_edges = prune(|id| graph.get(id), &mut c_with_dist, &r, &a);
        graph.overwirte_out_edges(&p, new_edges);
    }

    for grave_i in &cemetery {
        graph.overwirte_out_edges(grave_i, vec![]); // Backlinks are not defined in the original algorithm but should be deleted here.
    }

    for grave_i in &cemetery {
        graph.free(grave_i)
    }

    graph.clear_cemetery();
}

fn prune<P, F>(
    mut get: F,
    candidates: &mut Vec<(f32, u32)>,
    builder_r: &usize,
    builder_a: &f32,
) -> Vec<u32>
where
    P: PointInterface,
    F: FnMut(&u32) -> (P, Vec<u32>),
{
    let mut new_n_out = vec![];

    while let Some((first, rest)) = candidates.split_first() {
        let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
        new_n_out.push(pa);

        if new_n_out.len() == *builder_r {
            break;
        }
        *candidates = rest.to_vec();

        // if α · d(p*, p') <= d(p, p') then remove p' from v
        candidates.retain(|&(dist_xp_pd, pd)| {
            // let pa_point = &self.nodes[pa].p;
            // let pd_point = &self.nodes[pd].p;
            let (pa_point, _) = get(&pa);
            let (pd_point, _) = get(&pd);
            let dist_pa_pd = pa_point.distance(&pd_point);

            builder_a * dist_pa_pd > dist_xp_pd
        })
    }

    new_n_out
}

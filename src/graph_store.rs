use crate::storage::StorageTrait;
use crate::utils;

use anyhow::Result;

type StoreIndex = u32;
// type SectorIndex = u32;

#[derive(Clone)]
pub struct GraphStore<S: StorageTrait> {
    storage: S,
    // // wip: これらのパラメータは、ストレージのヘッダーに書き込んだ方がいい
    // num_vectors: usize,
    // vector_dim: usize,
    // max_edge_degrees: usize,
    // node_byte_size: usize,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct GraphHeader {
    start_id: u32,
    num_vectors: u32,
    vector_dim: u32,
    max_edge_degrees: u32,
    node_byte_size: u32,
}

impl<S> GraphStore<S>
where
    S: StorageTrait,
{
    pub fn new(num_vectors: usize, vector_dim: usize, max_edge_degrees: usize, storage: S) -> Self {
        let node_byte_size = utils::node_byte_size(vector_dim, max_edge_degrees);
        let header: GraphHeader = GraphHeader {
            start_id: 0,
            num_vectors: num_vectors as u32,
            vector_dim: vector_dim as u32,
            max_edge_degrees: max_edge_degrees as u32,
            node_byte_size: node_byte_size as u32,
        };

        let header_bytes = bincode::serialize(&header).unwrap();
        storage.write(0, &header_bytes);

        Self {
            storage,
            // num_vectors,
            // vector_dim,
            // max_edge_degrees,
            // node_byte_size,
        }
    }

    pub fn load(storage: S) -> Self {
        Self {
            storage,
            // num_vectors: header.num_vectors as usize,
            // vector_dim: header.vector_dim as usize,
            // max_edge_degrees: header.max_edge_degrees as usize,
            // node_byte_size: header.node_byte_size as usize,
        }
    }

    pub fn set_start_id(&self, start_id: u32) {
        let mut header: GraphHeader = self.read_header();
        header.start_id = start_id;
        self.write_header(header);
    }

    fn offset_from_store_index(&self, store_index: &StoreIndex) -> u64 {
        std::mem::size_of::<GraphHeader>() as u64
            + *store_index as u64 * self.node_byte_size() as u64
    }

    pub fn read_serialized_node(&self, store_index: &StoreIndex) -> Vec<u8> {
        let offset = self.offset_from_store_index(store_index);
        let mut bytes: Vec<u8> = vec![0; self.node_byte_size()];
        self.storage.read(offset, &mut bytes);
        bytes
    }

    pub fn read_node(&self, store_index: &StoreIndex) -> Result<(Vec<f32>, Vec<u32>)> {
        let bytes = self.read_serialized_node(store_index);

        let (vector, edges) =
            utils::deserialize_node(&bytes, self.vector_dim(), self.max_edge_degrees());

        Ok((vector.to_vec(), edges.to_vec()))
    }

    pub fn read_edges(&self, store_index: &StoreIndex) -> Result<Vec<u32>> {
        let (_, edges) = self.read_node(store_index)?;
        Ok(edges)
    }

    pub fn write_node(
        &self,
        store_index: &StoreIndex,
        vector: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()> {
        let bytes = utils::serialize_node(vector, edges);
        let offset = self.offset_from_store_index(store_index);
        self.storage.write(offset, &bytes);
        Ok(())
    }

    fn read_header(&self) -> GraphHeader {
        let mut bytes = vec![0u8; std::mem::size_of::<GraphHeader>()];
        self.storage.read(0, &mut bytes);
        let header: GraphHeader = bincode::deserialize(&bytes).unwrap();
        header
    }

    fn write_header(&self, header: GraphHeader) {
        let header_bytes = bincode::serialize(&header).unwrap();
        self.storage.write(0, &header_bytes);
    }

    pub fn num_vectors(&self) -> usize {
        self.read_header().num_vectors as usize
    }
    pub fn vector_dim(&self) -> usize {
        self.read_header().vector_dim as usize
    }
    pub fn max_edge_degrees(&self) -> usize {
        self.read_header().max_edge_degrees as usize
    }
    pub fn node_byte_size(&self) -> usize {
        self.read_header().node_byte_size as usize
    }
    pub fn start_id(&self) -> usize {
        self.read_header().start_id as usize
    }
    pub fn into_storage(self) -> S {
        self.storage
    }
}

pub struct EdgesIterator<'a, S: StorageTrait> {
    graph: &'a GraphStore<S>,
    index: StoreIndex,
    current_position: usize,
    edges: Vec<u32>,
}

impl<'a, S: StorageTrait> EdgesIterator<'a, S> {
    pub fn new(graph: &'a GraphStore<S>) -> Self {
        let edges = match graph.read_edges(&0) {
            Ok(edges) => edges,
            Err(_) => panic!(), // wip
        };

        EdgesIterator {
            graph,
            index: 0,
            current_position: 0,
            edges,
        }
    }
}

type EdgeIndex = u32;

impl<'a, S: StorageTrait> Iterator for EdgesIterator<'a, S> {
    type Item = std::result::Result<(EdgeIndex, u32), std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.edges.len() {
            // If the current position exceeds the length of the edge list, the next edge set is read
            self.current_position = 0;
            self.index += 1;

            if self.index >= self.graph.num_vectors() as u32 {
                return None;
            } else {
                match self.graph.read_edges(&self.index) {
                    Ok(new_edges) => {
                        if new_edges.is_empty() {
                            return None;
                        }
                        self.edges = new_edges;
                    }
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        )))
                    }
                }
            }
        }

        let result = (self.edges[self.current_position], (self.index) as u32);
        self.current_position += 1;
        Some(Ok(result))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bytesize::MB;

    use crate::storage::Storage;

    use super::GraphStore;
    // const SECTOR_BYTES_SIZE: usize = 96 * 4 + 70 * 4 + 4;

    #[test]
    fn write_and_read_node() {
        let storage =
            Storage::new_with_empty_file(Path::new("test_vectors/test.graph"), MB).unwrap();
        let graph_on_stroage = GraphStore::new(100, 96, 70 * 2, storage);

        graph_on_stroage
            .write_node(&10, &vec![1.0; 96], &vec![1; 140])
            .unwrap();

        let (p, e) = graph_on_stroage.read_node(&10).unwrap();
        println!("{:?}, {:?}", p, e);
        assert_eq!(p, vec![1.0; 96]);
        assert_eq!(e, vec![1; 140])
    }
}

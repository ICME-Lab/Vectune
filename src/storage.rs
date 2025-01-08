use anyhow::Result;
use memmap2::{MmapMut, MmapOptions};
use std::{fs::OpenOptions, path::Path, sync::Arc};

/* types */
pub trait StorageTrait {
    fn read(&self, offset: u64, dst: &mut [u8]);
    fn write(&self, offset: u64, src: &[u8]);
    // fn sector_byte_size(&self) -> usize;

    // todo grow file size
}

#[derive(Clone)]
pub struct Storage {
    mmap_arc: Arc<MmapMut>,
    // sector_byte_size: usize,
}

impl Storage {
    pub fn new_with_empty_file(
        path: &Path,
        file_byte_size: u64,
        // sector_byte_size: usize,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(file_byte_size)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap_arc: Arc::new(mmap),
            // sector_byte_size,
        })
    }

    pub fn load(path: &str, // , sector_byte_size: usize
    ) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        println!("memmap len: {}", mmap.len());
        Ok(Self {
            mmap_arc: Arc::new(mmap),
            // sector_byte_size,
        })
    }
}

impl StorageTrait for Storage {
    fn read(&self, offset: u64, dst: &mut [u8]) {
        dst.copy_from_slice(&self.mmap_arc[offset as usize..offset as usize + dst.len()])
    }

    fn write(&self, offset: u64, src: &[u8]) {
        assert!(offset as usize + src.len() <= self.mmap_arc.len());

        let mmap_ref = Arc::clone(&self.mmap_arc);
        unsafe {
            let dest = mmap_ref.as_ptr().add(offset as usize) as *mut u8;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dest, src.len());
        }
    }

    // fn sector_byte_size(&self) -> usize {
    //     self.sector_byte_size
    // }
}

#[cfg(test)]
mod tests {

    use std::path::Path;

    use crate::storage::{Storage, StorageTrait};
    use bytesize::MB;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    #[test]
    fn write_and_read_byte_to_storage() {
        let storage =
            Storage::new_with_empty_file(Path::new("test_vectors/test.graph"), MB).unwrap();
        let original_bytes: Vec<u8> = vec![1; 10];
        let offset = 100;
        storage.write(offset, &original_bytes);

        let mut result_bytes = vec![0; 10];
        storage.read(offset, &mut result_bytes);

        assert_eq!(original_bytes, result_bytes);
    }

    #[test]
    fn par_write_byte_to_storage() {
        let storage =
            Storage::new_with_empty_file(Path::new("test_vectors/test.graph"), MB).unwrap();

        (0..9).into_par_iter().for_each(|i| {
            let offset = i * 10;
            let original_bytes: Vec<u8> = vec![i as u8; 10];
            storage.write(offset, &original_bytes)
        });

        (0..9).into_par_iter().for_each(|i| {
            let offset = i * 10;
            let original_bytes: Vec<u8> = vec![i as u8; 10];
            let mut result_bytes: Vec<u8> = vec![0; 10];
            storage.read(offset, &mut result_bytes);
            assert_eq!(original_bytes, result_bytes);
        });
    }
}

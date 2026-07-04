use crate::FrameAllocator;
use crate::Heap;
use crate::LockedHeapWithRescue;
use crate::linked_list;
use core::alloc::GlobalAlloc;
use core::alloc::Layout;
use core::mem::size_of;
use core::ptr::NonNull;
#[cfg(feature = "metadata_heap")]
use crate::MetadataHeap;

#[test]
fn test_linked_list() {
    let mut value1: usize = 0;
    let mut value2: usize = 0;
    let mut value3: usize = 0;
    let mut list = linked_list::LinkedList::new();
    unsafe {
        list.push(&mut value1 as *mut usize);
        list.push(&mut value2 as *mut usize);
        list.push(&mut value3 as *mut usize);
    }

    // Test links
    assert_eq!(value3, &value2 as *const usize as usize);
    assert_eq!(value2, &value1 as *const usize as usize);
    assert_eq!(value1, 0);

    // Test iter
    let mut iter = list.iter();
    assert_eq!(iter.next(), Some(&mut value3 as *mut usize));
    assert_eq!(iter.next(), Some(&mut value2 as *mut usize));
    assert_eq!(iter.next(), Some(&mut value1 as *mut usize));
    assert_eq!(iter.next(), None);

    // Test iter_mut

    let mut iter_mut = list.iter_mut();
    assert_eq!(iter_mut.next().unwrap().pop(), &mut value3 as *mut usize);

    // Test pop
    assert_eq!(list.pop(), Some(&mut value2 as *mut usize));
    assert_eq!(list.pop(), Some(&mut value1 as *mut usize));
    assert_eq!(list.pop(), None);
}

#[test]
fn test_empty_heap() {
    let mut heap = Heap::<32>::new();
    assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()).is_err());
}

#[test]
fn test_heap_add() {
    let mut heap = Heap::<32>::new();
    assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()).is_err());

    let space: [usize; 100] = [0; 100];
    unsafe {
        heap.add_to_heap(space.as_ptr() as usize, space.as_ptr().add(100) as usize);
    }
    let addr = heap.alloc(Layout::from_size_align(1, 1).unwrap());
    assert!(addr.is_ok());
}

#[test]
fn test_heap_add_large() {
    // Max size of block is 2^7 == 128 bytes
    let mut heap = Heap::<8>::new();
    assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()).is_err());

    // 512 bytes of space
    let space: [u8; 512] = [0; 512];
    unsafe {
        heap.add_to_heap(space.as_ptr() as usize, space.as_ptr().add(512) as usize);
    }
    let addr = heap.alloc(Layout::from_size_align(1, 1).unwrap());
    assert!(addr.is_ok());
}

#[test]
fn test_heap_oom() {
    let mut heap = Heap::<32>::new();
    let space: [usize; 100] = [0; 100];
    unsafe {
        heap.add_to_heap(space.as_ptr() as usize, space.as_ptr().add(100) as usize);
    }

    assert!(
        heap.alloc(Layout::from_size_align(100 * size_of::<usize>(), 1).unwrap())
            .is_err()
    );
    assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()).is_ok());
}

#[test]
fn test_heap_oom_rescue() {
    const SPACE_SIZE: usize = 100;
    static mut SPACE: [usize; 100] = [0; SPACE_SIZE];
    let heap = LockedHeapWithRescue::new(|heap: &mut Heap<32>, _layout: &Layout| unsafe {
        heap.init(&raw mut SPACE as usize, SPACE_SIZE);
    });

    unsafe {
        assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()) as usize != 0);
    }
}

#[test]
fn test_heap_alloc_and_free() {
    let mut heap = Heap::<32>::new();
    assert!(heap.alloc(Layout::from_size_align(1, 1).unwrap()).is_err());

    let space: [usize; 100] = [0; 100];
    unsafe {
        heap.add_to_heap(space.as_ptr() as usize, space.as_ptr().add(100) as usize);
    }
    for _ in 0..100 {
        let addr = heap.alloc(Layout::from_size_align(1, 1).unwrap()).unwrap();
        unsafe {
            heap.dealloc(addr, Layout::from_size_align(1, 1).unwrap());
        }
    }
}

#[test]
fn test_empty_frame_allocator() {
    let mut frame = FrameAllocator::<32>::new();
    assert!(frame.alloc(1).is_none());
}

#[test]
fn test_frame_allocator_add() {
    let mut frame = FrameAllocator::<32>::new();
    assert!(frame.alloc(1).is_none());

    frame.insert(0..3);
    let num = frame.alloc(1);
    assert_eq!(num, Some(2));
    let num = frame.alloc(2);
    assert_eq!(num, Some(0));
    assert!(frame.alloc(1).is_none());
    assert!(frame.alloc(2).is_none());
}

#[test]
fn test_frame_allocator_add_from_zero_keeps_large_block() {
    let mut frame = FrameAllocator::<7>::new();

    frame.add_frame(0, 64);

    assert_eq!(frame.alloc(64), Some(0));
}

#[test]
fn test_frame_allocator_allocate_large() {
    let mut frame = FrameAllocator::<32>::new();
    assert_eq!(frame.alloc(10_000_000_000), None);
}

#[test]
fn test_frame_allocator_add_large_size_split() {
    let mut frame = FrameAllocator::<32>::new();

    frame.insert(0..10_000_000_000);

    assert_eq!(frame.alloc(0x8000_0001), None);
    assert_eq!(frame.alloc(0x8000_0000), Some(0));
    assert_eq!(frame.alloc(0x8000_0000), Some(0x8000_0000));
}

#[test]
fn test_frame_allocator_add_large_size() {
    let mut frame = FrameAllocator::<33>::new();

    frame.insert(0..10_000_000_000);
    assert_eq!(frame.alloc(0x8000_0001), Some(0));
}

#[test]
fn test_frame_allocator_alloc_and_free() {
    let mut frame = FrameAllocator::<32>::new();
    assert!(frame.alloc(1).is_none());

    frame.add_frame(0, 1024);
    for _ in 0..100 {
        let addr = frame.alloc(512).unwrap();
        frame.dealloc(addr, 512);
    }
}

#[test]
fn test_frame_allocator_alloc_and_free_complex() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(100, 1024);
    for _ in 0..10 {
        let addr = frame.alloc(1).unwrap();
        frame.dealloc(addr, 1);
    }
    let addr1 = frame.alloc(1).unwrap();
    let addr2 = frame.alloc(1).unwrap();
    assert_ne!(addr1, addr2);
}

#[test]
fn test_frame_allocator_aligned() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(1, 64);
    assert_eq!(
        frame.alloc_aligned(Layout::from_size_align(2, 4).unwrap()),
        Some(4)
    );
    assert_eq!(
        frame.alloc_aligned(Layout::from_size_align(2, 2).unwrap()),
        Some(2)
    );
    assert_eq!(
        frame.alloc_aligned(Layout::from_size_align(2, 1).unwrap()),
        Some(8)
    );
    assert_eq!(
        frame.alloc_aligned(Layout::from_size_align(1, 16).unwrap()),
        Some(16)
    );
}

#[test]
fn test_frame_allocator_merge_final_order() {
    let mut frame = FrameAllocator::<2>::new();
    frame.add_frame(0, 4);

    let first = frame.alloc(2).unwrap();
    let second = frame.alloc(2).unwrap();

    frame.dealloc(first, 2);
    frame.dealloc(second, 2);

    assert_eq!(frame.alloc(2), Some(0));
}

#[test]
fn test_heap_merge_final_order() {
    const NUM_ORDERS: usize = 5;

    let backing_size = 1 << NUM_ORDERS;
    let backing_layout = Layout::from_size_align(backing_size, backing_size).unwrap();

    // create a new heap with 5 orders
    let mut heap = Heap::<NUM_ORDERS>::new();

    // allocate host memory for use by heap
    let backing_allocation = unsafe { std::alloc::alloc(backing_layout) };

    let start = backing_allocation as usize;
    let middle = unsafe { backing_allocation.add(backing_size / 2) } as usize;
    let end = unsafe { backing_allocation.add(backing_size) } as usize;

    // add two contiguous ranges of memory
    unsafe { heap.add_to_heap(start, middle) };
    unsafe { heap.add_to_heap(middle, end) };

    // NUM_ORDERS - 1 is the maximum order of the heap
    let layout = Layout::from_size_align(1 << (NUM_ORDERS - 1), 1).unwrap();

    // allocation should succeed, using one of the added ranges
    let alloc = heap.alloc(layout).unwrap();

    // deallocation should not attempt to merge the two contiguous ranges as the next order does not exist
    unsafe {
        heap.dealloc(alloc, layout);
    }
}

#[test]
fn test_frame_allocator_alloc_at_basic() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 4);
    assert_eq!(frame.alloc_at(0, 4), Some(0));
    assert!(frame.alloc(1).is_none());
}

#[test]
fn test_frame_allocator_alloc_at_split() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 8);
    // Alloc 2 frames at address 2 (requires splitting the order-3 block)
    assert_eq!(frame.alloc_at(2, 2), Some(2));
    // Remaining: [0..2) at order 1, [4..8) at order 2
    assert_eq!(frame.alloc(2), Some(0));
    assert_eq!(frame.alloc(4), Some(4));
    assert!(frame.alloc(1).is_none());
}

#[test]
fn test_frame_allocator_alloc_at_unavailable() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 8);
    assert_eq!(frame.alloc(4), Some(0));
    // [0..4) is allocated, try to alloc_at within it
    assert_eq!(frame.alloc_at(0, 2), None);
    assert_eq!(frame.alloc_at(2, 2), None);
}

#[test]
fn test_frame_allocator_alloc_at_misaligned() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 16);
    // 4 frames at address 3: not aligned to 4
    assert_eq!(frame.alloc_at(3, 4), None);
    // 2 frames at address 1: not aligned to 2
    assert_eq!(frame.alloc_at(1, 2), None);
    // 1 frame at address 1: aligned to 1, should work
    assert_eq!(frame.alloc_at(1, 1), Some(1));
}

#[test]
fn test_frame_allocator_alloc_at_then_dealloc() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 16);
    assert_eq!(frame.alloc_at(4, 4), Some(4));
    frame.dealloc(4, 4);
    // Buddies should merge back; full 16-frame alloc should succeed
    assert_eq!(frame.alloc(16), Some(0));
}

#[test]
fn test_frame_allocator_alloc_at_outside_range() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 8);
    assert_eq!(frame.alloc_at(16, 2), None);
}

#[test]
fn test_frame_allocator_alloc_at_multiple() {
    let mut frame = FrameAllocator::<32>::new();
    frame.add_frame(0, 16);
    assert_eq!(frame.alloc_at(0, 4), Some(0));
    assert_eq!(frame.alloc_at(4, 4), Some(4));
    assert_eq!(frame.alloc_at(8, 4), Some(8));
    assert_eq!(frame.alloc_at(12, 4), Some(12));
    assert!(frame.alloc(1).is_none());
}

// ============================================================================
// HeapLike trait — shared conformance test infrastructure
// ============================================================================

#[cfg(feature = "metadata_heap")]
use crate::InitError;

use core::cmp::min;

/// A page-aligned backing array for heap tests.
#[repr(align(4096))]
struct PageAligned<const N: usize>([u8; N]);

/// Trait abstracting the common API between [`Heap`] and [`MetadataHeap`]
/// so that conformance tests can be written once and exercised against
/// both implementations.
trait HeapLike {
    /// Initialise with `[start, start + size)`.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`Heap::init`] / [`MetadataHeap::try_init`].
    unsafe fn init_region(&mut self, start: usize, size: usize) -> Result<(), ()>;

    /// Allocate a block satisfying `layout`.
    fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()>;

    /// Deallocate a block previously returned by [`alloc`](HeapLike::alloc).
    ///
    /// # Safety
    ///
    /// `ptr` and `layout` must exactly match a previous successful allocation
    /// from this heap instance, and not already have been deallocated.
    unsafe fn dealloc_region(&mut self, ptr: NonNull<u8>, layout: Layout);

    /// Total bytes requested by the user (sum of `layout.size()`).
    fn stats_alloc_user(&self) -> usize;

    /// Total bytes actually carved from the heap (power-of-two rounded).
    fn stats_alloc_actual(&self) -> usize;

    /// Total usable bytes in the data region.
    fn stats_total_bytes(&self) -> usize;
}

impl HeapLike for Heap<32> {
    unsafe fn init_region(&mut self, start: usize, size: usize) -> Result<(), ()> {
        unsafe {
            self.init(start, size);
        }
        Ok(())
    }

    fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        self.alloc(layout)
    }

    unsafe fn dealloc_region(&mut self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            self.dealloc(ptr, layout);
        }
    }

    fn stats_alloc_user(&self) -> usize {
        self.stats_alloc_user()
    }
    fn stats_alloc_actual(&self) -> usize {
        self.stats_alloc_actual()
    }
    fn stats_total_bytes(&self) -> usize {
        self.stats_total_bytes()
    }
}

#[cfg(feature = "metadata_heap")]
impl HeapLike for MetadataHeap<32, 3> {
    unsafe fn init_region(&mut self, start: usize, size: usize) -> Result<(), ()> {
        unsafe { self.try_init(start, size).map_err(|_| ()) }
    }

    fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        self.alloc(layout)
    }

    unsafe fn dealloc_region(&mut self, ptr: NonNull<u8>, layout: Layout) {
        unsafe {
            self.dealloc(ptr, layout);
        }
    }

    fn stats_alloc_user(&self) -> usize {
        self.stats_alloc_user()
    }
    fn stats_alloc_actual(&self) -> usize {
        self.stats_alloc_actual()
    }
    fn stats_total_bytes(&self) -> usize {
        self.stats_total_bytes()
    }
}

// ============================================================================
// Shared conformance helpers
// ============================================================================

/// Alloc 64 bytes, dealloc, verify stats reset.
fn conformance_alloc_dealloc_round_trip<T: HeapLike>(mut heap: T, start: usize, size: usize) {
    unsafe {
        heap.init_region(start, size).unwrap();
    }
    assert!(heap.stats_total_bytes() > 0);
    let layout = Layout::from_size_align(64, 1).unwrap();
    let ptr = heap.alloc(layout).unwrap();
    assert!(heap.stats_alloc_user() > 0);
    assert!(heap.stats_alloc_actual() > 0);
    unsafe {
        heap.dealloc_region(ptr, layout);
    }
    assert_eq!(heap.stats_alloc_user(), 0);
    assert_eq!(heap.stats_alloc_actual(), 0);
}

/// Alloc 8, 64, 256, 1024 bytes, dealloc all, stats=0.
fn conformance_alloc_multiple_sizes<T: HeapLike>(mut heap: T, start: usize, size: usize) {
    unsafe {
        heap.init_region(start, size).unwrap();
    }

    let sizes: [usize; 4] = [8, 64, 256, 1024];
    let mut allocs: [Option<(NonNull<u8>, Layout)>; 4] =
        [const { None }; 4];

    let mut expected_user: usize = 0;
    for (i, &sz) in sizes.iter().enumerate() {
        let layout = Layout::from_size_align(sz, 1).unwrap();
        let ptr = heap.alloc(layout).unwrap();
        expected_user += sz;
        assert_eq!(heap.stats_alloc_user(), expected_user);
        allocs[i] = Some((ptr, layout));
    }

    for slot in &allocs {
        let (ptr, layout) = slot.unwrap();
        unsafe {
            heap.dealloc_region(ptr, layout);
        }
    }

    assert_eq!(heap.stats_alloc_user(), 0);
    assert_eq!(heap.stats_alloc_actual(), 0);
}

/// Exhaust the heap, verify next alloc fails.
fn conformance_oom<T: HeapLike>(mut heap: T, start: usize, size: usize) {
    unsafe {
        heap.init_region(start, size).unwrap();
    }

    let layout = Layout::from_size_align(1, 1).unwrap();
    let mut count: usize = 0;
    let mut allocs: std::vec::Vec<NonNull<u8>> = std::vec::Vec::new();

    while let Ok(ptr) = heap.alloc(layout) {
        count += 1;
        allocs.push(ptr);
    }

    assert!(count > 0, "should have allocated at least one block");
    assert!(heap.alloc(layout).is_err());

    // Cleanup
    for ptr in allocs {
        unsafe {
            heap.dealloc_region(ptr, layout);
        }
    }
}

/// Alloc two buddies, free both, verify they merge (next alloc of double size succeeds).
fn conformance_merge_after_free<T: HeapLike>(mut heap: T, start: usize, size: usize) {
    unsafe {
        heap.init_region(start, size).unwrap();
    }

    let layout_64 = Layout::from_size_align(64, 1).unwrap();
    let ptr1 = heap.alloc(layout_64).unwrap();
    let ptr2 = heap.alloc(layout_64).unwrap();

    let addr1 = ptr1.as_ptr() as usize;
    let addr2 = ptr2.as_ptr() as usize;

    // In a buddy system, two consecutive allocs of the same order
    // from a fresh heap should be buddies.
    assert_eq!(
        addr1 ^ addr2,
        64,
        "expected buddy addresses differing by 64"
    );

    unsafe {
        heap.dealloc_region(ptr1, layout_64);
        heap.dealloc_region(ptr2, layout_64);
    }

    // Now a 128-byte alloc should succeed (the two buddies merged).
    let layout_128 = Layout::from_size_align(128, 1).unwrap();
    let ptr3 = heap.alloc(layout_128).unwrap();

    // The returned pointer should be the lower buddy address.
    assert_eq!(ptr3.as_ptr() as usize, min(addr1, addr2));

    unsafe {
        heap.dealloc_region(ptr3, layout_128);
    }
}

// ============================================================================
// Conformance tests — Heap<32>
// ============================================================================

#[test]
fn test_conformance_alloc_dealloc_round_trip_heap() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    conformance_alloc_dealloc_round_trip(Heap::<32>::empty(), space.0.as_ptr() as usize, N);
}

#[test]
fn test_conformance_alloc_multiple_sizes_heap() {
    const N: usize = 32768;
    let space = PageAligned::<N>([0; N]);
    conformance_alloc_multiple_sizes(Heap::<32>::empty(), space.0.as_ptr() as usize, N);
}

#[test]
fn test_conformance_oom_heap() {
    const N: usize = 256;
    let space = PageAligned::<N>([0; N]);
    conformance_oom(Heap::<32>::empty(), space.0.as_ptr() as usize, N);
}

#[test]
fn test_conformance_merge_after_free_heap() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    conformance_merge_after_free(Heap::<32>::empty(), space.0.as_ptr() as usize, N);
}

// ============================================================================
// Conformance tests — MetadataHeap<32, 3>
// ============================================================================

#[cfg(feature = "metadata_heap")]
#[test]
fn test_conformance_alloc_dealloc_round_trip_mh() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    conformance_alloc_dealloc_round_trip(
        MetadataHeap::<32, 3>::empty(),
        space.0.as_ptr() as usize,
        N,
    );
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_conformance_alloc_multiple_sizes_mh() {
    const N: usize = 32768;
    let space = PageAligned::<N>([0; N]);
    conformance_alloc_multiple_sizes(
        MetadataHeap::<32, 3>::empty(),
        space.0.as_ptr() as usize,
        N,
    );
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_conformance_oom_mh() {
    const N: usize = 256;
    let space = PageAligned::<N>([0; N]);
    conformance_oom(
        MetadataHeap::<32, 3>::empty(),
        space.0.as_ptr() as usize,
        N,
    );
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_conformance_merge_after_free_mh() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    conformance_merge_after_free(
        MetadataHeap::<32, 3>::empty(),
        space.0.as_ptr() as usize,
        N,
    );
}

// ============================================================================
// MetadataHeap-specific tests
// ============================================================================

#[cfg(feature = "metadata_heap")]
#[test]
fn test_metadata_heap_init_too_small() {
    const N: usize = 7;
    let space = PageAligned::<N>([0; N]);
    let start = space.0.as_ptr() as usize;
    let mut heap = MetadataHeap::<32, 3>::empty();
    let result = unsafe { heap.try_init(start, N) };
    match result {
        Err(InitError::TooSmall) => {} // expected
        other => panic!("expected Err(TooSmall), got {other:?}"),
    }
    // Stats must remain unchanged.
    assert_eq!(heap.stats_alloc_user(), 0);
    assert_eq!(heap.stats_alloc_actual(), 0);
    assert_eq!(heap.stats_total_bytes(), 0);
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_metadata_heap_stats_use_original_order() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    let start = space.0.as_ptr() as usize;
    let mut heap = MetadataHeap::<32, 3>::empty();
    unsafe {
        heap.try_init(start, N).unwrap();
    }

    // 24 bytes rounds up to 32 (order 5).
    let layout = Layout::from_size_align(24, 1).unwrap();
    let ptr = heap.alloc(layout).unwrap();

    // stats_alloc_actual should be 32 (2^5), not 24.
    assert_eq!(heap.stats_alloc_actual(), 32);
    assert_eq!(heap.stats_alloc_user(), 24);

    unsafe {
        heap.dealloc(ptr, layout);
    }
    assert_eq!(heap.stats_alloc_actual(), 0);
    assert_eq!(heap.stats_alloc_user(), 0);
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_metadata_heap_absolute_alignment() {
    const N: usize = 32768;
    let space = PageAligned::<N>([0; N]);
    // Shift start by 1 to create an unaligned region origin.
    let start = (space.0.as_ptr() as usize) + 1;
    let size = N - 1;

    let mut heap = MetadataHeap::<32, 3>::empty();
    unsafe {
        heap.try_init(start, size).unwrap();
    }

    let layout = Layout::from_size_align(1, 4096).unwrap();
    let ptr = heap.alloc(layout).unwrap();
    assert_eq!(
        ptr.as_ptr() as usize & 4095,
        0,
        "returned pointer should be 4096-aligned"
    );

    unsafe {
        heap.dealloc(ptr, layout);
    }
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_metadata_heap_min_order_controls_min_block() {
    // MIN_ORDER=3 → smallest block 8 bytes (order 3).
    {
        const N: usize = 4096;
        let space = PageAligned::<N>([0; N]);
        let start = space.0.as_ptr() as usize;
        let mut heap = MetadataHeap::<16, 3>::empty();
        unsafe {
            heap.try_init(start, N).unwrap();
        }

        let layout = Layout::from_size_align(1, 1).unwrap();
        let ptr = heap.alloc(layout).unwrap();
        assert_eq!(heap.stats_alloc_actual(), 8); // 2^3
        unsafe {
            heap.dealloc(ptr, layout);
        }
    }

    // MIN_ORDER=6 → smallest block 64 bytes (order 6).
    {
        const N: usize = 4096;
        let space = PageAligned::<N>([0; N]);
        let start = space.0.as_ptr() as usize;
        let mut heap = MetadataHeap::<16, 6>::empty();
        unsafe {
            heap.try_init(start, N).unwrap();
        }

        let layout = Layout::from_size_align(1, 1).unwrap();
        let ptr = heap.alloc(layout).unwrap();
        assert_eq!(heap.stats_alloc_actual(), 64); // 2^6
        unsafe {
            heap.dealloc(ptr, layout);
        }
    }
}

#[cfg(all(feature = "metadata_heap", debug_assertions))]
#[test]
#[should_panic(expected = "dealloc on non-Used block head")]
fn test_metadata_heap_debug_double_free_panics() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    let start = space.0.as_ptr() as usize;
    let mut heap = MetadataHeap::<32, 3>::empty();
    unsafe {
        heap.try_init(start, N).unwrap();
    }

    let layout = Layout::from_size_align(16, 1).unwrap();
    let ptr = heap.alloc(layout).unwrap();
    unsafe {
        heap.dealloc(ptr, layout);
    }
    // Double-free should panic in debug builds.
    unsafe {
        heap.dealloc(ptr, layout);
    }
}

#[cfg(feature = "metadata_heap")]
#[test]
fn test_metadata_heap_both_buddies_merge() {
    const N: usize = 4096;
    let space = PageAligned::<N>([0; N]);
    let start = space.0.as_ptr() as usize;
    let mut heap = MetadataHeap::<32, 3>::empty();
    unsafe {
        heap.try_init(start, N).unwrap();
    }

    let layout_64 = Layout::from_size_align(64, 1).unwrap();
    let ptr1 = heap.alloc(layout_64).unwrap();
    let ptr2 = heap.alloc(layout_64).unwrap();

    let addr1 = ptr1.as_ptr() as usize;
    let addr2 = ptr2.as_ptr() as usize;

    // Verify they are buddies (addresses differ by block size).
    assert_eq!(
        addr1 ^ addr2,
        64,
        "expected buddy addresses differing by 64"
    );

    unsafe {
        heap.dealloc(ptr1, layout_64);
        heap.dealloc(ptr2, layout_64);
    }

    // After merging, a 128-byte alloc should succeed.
    let layout_128 = Layout::from_size_align(128, 1).unwrap();
    let ptr3 = heap.alloc(layout_128).unwrap();

    // The return value should be the lower buddy address, 128-aligned.
    assert_eq!(ptr3.as_ptr() as usize, min(addr1, addr2));
    assert_eq!(
        ptr3.as_ptr() as usize & 127,
        0,
        "merged block should be 128-aligned"
    );

    unsafe {
        heap.dealloc(ptr3, layout_128);
    }
}

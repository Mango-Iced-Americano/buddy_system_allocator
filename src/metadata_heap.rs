//! A buddy allocator with explicit per-unit metadata enabling O(1) buddy checks.
//!
//! Unlike the linked-list-based [`Heap`](crate::Heap), this allocator stores a
//! [`BlockMeta`] per minimal allocatable unit (`1 << MIN_ORDER` bytes) in a
//! contiguous metadata array carved from the front of the managed region.
//! This allows O(1) buddy-state queries during deallocation, avoiding the
//! linear scan that the linked-list variant requires.

use core::alloc::Layout;
use core::cmp::{max, min};
use core::fmt;
use core::mem::{align_of, size_of};
use core::ops::Deref;
use core::ptr::NonNull;

#[cfg(feature = "use_spin")]
use spin::Mutex;

use crate::prev_power_of_two;

// ---------------------------------------------------------------------------
// Compile-time sanity checks
// ---------------------------------------------------------------------------

const _: () = assert!(
    size_of::<usize>() <= 1usize << (usize::BITS as usize - 1),
    "usize sanity"
);

// ---------------------------------------------------------------------------
// Constants & types
// ---------------------------------------------------------------------------

/// Sentinel value for "no predecessor / no successor" in the doubly-linked
/// free-list embedded in [`BlockMeta`].
const NIL: usize = usize::MAX;

/// The state of a single minimal-unit block in the buddy system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum BlockState {
    Reserved = 0,
    Free = 1,
    Used = 2,
}

/// Per-unit metadata stored in the metadata array.
///
/// - `state`:    current [`BlockState`] of this unit.
/// - `order`:    meaningful only for `Free` and `Used` heads.
/// - `prev`/`next`: doubly-linked free-list pointers (valid only when `Free`).
#[derive(Clone, Copy, Debug)]
struct BlockMeta {
    state: BlockState,
    order: u8,
    prev: usize,
    next: usize,
}

impl BlockMeta {
    const fn reserved() -> Self {
        Self {
            state: BlockState::Reserved,
            order: 0,
            prev: NIL,
            next: NIL,
        }
    }
}

/// Head + length of one free-list (all blocks of a given order).
#[derive(Clone, Copy, Debug)]
struct FreeArea {
    head: usize,
    len: usize,
}

impl FreeArea {
    const EMPTY: Self = Self { head: NIL, len: 0 };
}

// ---------------------------------------------------------------------------
// Initialisation error
// ---------------------------------------------------------------------------

/// Error returned by [`MetadataHeap::try_init`].
#[derive(Debug)]
pub enum InitError {
    /// Arithmetic overflow during region computation.
    Overflow,
    /// The supplied region is too small for even one allocation unit plus its metadata.
    TooSmall,
}

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

/// Round `addr` up to the next multiple of `align`.
/// Returns `None` on overflow.
#[inline]
fn align_up(addr: usize, align: usize) -> Option<usize> {
    if align == 0 {
        return None;
    }
    Some(addr.checked_add(align - 1)? & !(align - 1))
}

/// Round `addr` down to the previous multiple of `align`.
#[inline]
fn align_down(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
}

/// System-allocator-compatible error for page-granular operations.
#[derive(Debug)]
pub enum AllocError {
    /// No sufficiently-sized free block available.
    NoMemory,
    /// The requested order is outside `[MIN_ORDER, ORDER)`.
    InvalidOrder,
}

/// Order of a page-granular allocation (block size = `1 << order.0`).
#[derive(Clone, Copy, Debug)]
pub struct PageOrder(pub u8);

/// A handle returned by [`MetadataHeap::alloc_pages`].
#[derive(Clone, Copy, Debug)]
pub struct PageRun {
    /// Base address of the allocated block.
    pub base: NonNull<u8>,
    /// Order of the block.
    pub order: PageOrder,
}

// ---------------------------------------------------------------------------
// The allocator
// ---------------------------------------------------------------------------

/// A buddy allocator with explicit per-unit metadata.
///
/// # Type parameters
///
/// - `ORDER`:     maximum order (the free-area array has `ORDER` slots,
///                supporting blocks up to order `ORDER - 1`).
/// - `MIN_ORDER`: smallest allocatable block order (e.g. `3` for 8 B,
///                `12` for 4 KiB).  Must satisfy `MIN_ORDER < ORDER` and
///                `ORDER ≤ 255`.
///
/// # Metadata layout
///
/// ```text
/// ┌────────────── metadata ──────────────┬─────── data region ───────┐
/// │ [BlockMeta; n]                       │  1<<MIN_ORDER per unit     │
/// └─ start                          heap_start                  heap_end
/// ```
///
/// The metadata array lives at the front of the managed region.  Each
/// [`BlockMeta`] describes one minimal unit (`1 << MIN_ORDER` bytes) of the
/// data region.  Block heads are tracked with their actual order; all
/// interior units of a multi-unit block carry `BlockState::Reserved`.
pub struct MetadataHeap<const ORDER: usize, const MIN_ORDER: usize> {
    /// Per-order free-list heads + lengths.
    free_area: [FreeArea; ORDER],

    /// Start of the metadata array (beginning of the managed region).
    start: usize,
    /// End of the metadata array.
    end: usize,

    /// Start of the data region (after metadata carve-out).
    heap_start: usize,
    /// End of the data region.
    heap_end: usize,

    /// Raw pointer to the metadata array.
    meta: *mut BlockMeta,
    /// Number of metadata entries (= number of minimal units).
    meta_len: usize,

    // Statistics (same naming as `Heap`)
    user: usize,
    allocated: usize,
    total: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl<const ORDER: usize, const MIN_ORDER: usize> MetadataHeap<ORDER, MIN_ORDER> {
    /// Create an uninitialised, empty allocator.
    ///
    /// Call [`try_init`](Self::try_init) before using any allocation method.
    pub const fn new() -> Self {
        MetadataHeap {
            free_area: [FreeArea::EMPTY; ORDER],
            start: 0,
            end: 0,
            heap_start: 0,
            heap_end: 0,
            meta: core::ptr::null_mut(),
            meta_len: 0,
            user: 0,
            allocated: 0,
            total: 0,
        }
    }

    /// Alias for [`new`](Self::new).
    pub const fn empty() -> Self {
        Self::new()
    }

    /// Initialise the allocator with a contiguous memory region.
    ///
    /// The region `[start, start + size)` is divided into two parts:
    /// a metadata array at the front, and the data region after it.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the memory range `[start, start + size)`
    /// is valid, writable, not managed by any other allocator, and remains
    /// available for the lifetime of this `MetadataHeap`.
    ///
    /// # Errors
    ///
    /// Returns [`InitError::TooSmall`] if the region cannot accommodate at
    /// least one allocation unit with its metadata, or
    /// [`InitError::Overflow`] on arithmetic overflow.
    pub unsafe fn try_init(&mut self, start: usize, size: usize) -> Result<(), InitError> {
        // ---- 1. Validate fundamental constraints ---------------------------
        if MIN_ORDER >= ORDER {
            return Err(InitError::TooSmall);
        }
        if ORDER > u8::MAX as usize {
            return Err(InitError::TooSmall);
        }

        let unit: usize = 1 << MIN_ORDER;
        if unit < size_of::<usize>() {
            return Err(InitError::TooSmall);
        }

        let meta_unit: usize = size_of::<BlockMeta>();
        let meta_align: usize = align_of::<BlockMeta>();

        // ---- 2. Align region bounds ---------------------------------------
        let meta_start: usize =
            align_up(start, meta_align).ok_or(InitError::Overflow)?;
        let region_end: usize = start
            .checked_add(size)
            .ok_or(InitError::Overflow)?;
        let data_end: usize = align_down(region_end, unit);

        if meta_start >= data_end {
            return Err(InitError::TooSmall);
        }

        // ---- 3. Binary-search the largest feasible n ----------------------
        // Constraint for n units:
        //   meta_bytes     = n * meta_unit
        //   heap_start     = align_up(meta_start + meta_bytes, unit)
        //   data_bytes     = n * unit
        //   need: heap_start + data_bytes ≤ data_end
        //
        // Loose upper bound for n:
        let max_search: usize = (size / (meta_unit + unit))
            .min(data_end.saturating_sub(meta_start) / meta_unit);

        if max_search == 0 {
            return Err(InitError::TooSmall);
        }

        let mut lo: usize = 0;
        let mut hi: usize = max_search + 1;

        while lo + 1 < hi {
            let mid: usize = lo + (hi - lo) / 2;

            let meta_end: usize = meta_start
                .checked_add(mid.checked_mul(meta_unit).ok_or(InitError::Overflow)?)
                .ok_or(InitError::Overflow)?;

            let candidate_heap_start: usize =
                match align_up(meta_end, unit) {
                    Some(h) => h,
                    None => {
                        hi = mid;
                        continue;
                    }
                };

            let candidate_needed: Option<usize> =
                candidate_heap_start.checked_add(mid.checked_mul(unit).ok_or(InitError::Overflow)?);

            if candidate_needed.map_or(true, |d| d > data_end) {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        if lo == 0 {
            return Err(InitError::TooSmall);
        }

        let n: usize = lo;
        let meta_bytes: usize = n * meta_unit;
        let heap_start: usize = align_up(meta_start + meta_bytes, unit)
            .ok_or(InitError::Overflow)?;
        let heap_end: usize = heap_start
            .checked_add(n.checked_mul(unit).ok_or(InitError::Overflow)?)
            .ok_or(InitError::Overflow)?;

        // ---- 4. Initialise metadata array to Reserved ---------------------
        self.meta = meta_start as *mut BlockMeta;
        self.meta_len = n;
        self.start = meta_start;
        self.end = meta_start + meta_bytes;
        self.heap_start = heap_start;
        self.heap_end = heap_end;

        for i in 0..n {
            unsafe {
                self.meta.add(i).write(BlockMeta::reserved());
            }
        }

        // ---- 5. Populate free_area with naturally-aligned blocks ----------
        let mut current: usize = heap_start;
        while current + unit <= heap_end {
            let lowbit: usize = current & current.wrapping_neg();
            let mut block_size: usize =
                min(lowbit, prev_power_of_two(heap_end - current));

            let mut order: usize = block_size.trailing_zeros() as usize;
            if order > ORDER - 1 {
                order = ORDER - 1;
                block_size = 1 << order;
            }

            if order >= MIN_ORDER {
                let idx: usize = self.addr_to_idx(current);
                unsafe {
                    self.push_free(idx, order);
                }
            }

            current += block_size;
        }

        // ---- 6. Set statistics --------------------------------------------
        self.total = heap_end - heap_start;
        self.user = 0;
        self.allocated = 0;

        Ok(())
    }

    /// Allocate a block satisfying `layout`.
    ///
    /// Returns `Err(())` when no suitably-sized free block is available.
    pub fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, ()> {
        let target_order: usize = self.layout_order(layout)?;

        // Find the first non-empty free-list at order ≥ target_order.
        for i in target_order..ORDER {
            if self.free_area[i].len == 0 {
                continue;
            }

            let idx: usize = self.pop_free(i).unwrap();
            let current_addr: usize = self.idx_to_addr(idx);

            // Split down to target_order, pushing right halves as Free.
            for j in (target_order + 1..=i).rev() {
                let buddy_addr: usize = current_addr + (1 << (j - 1));
                let buddy_idx: usize = self.addr_to_idx(buddy_addr);
                unsafe {
                    self.push_free(buddy_idx, j - 1);
                }
                // current_addr stays at the left half
            }

            // Mark final block as Used.
            unsafe {
                (*self.meta.add(idx)).state = BlockState::Used;
                (*self.meta.add(idx)).order = target_order as u8;
            }

            self.user += layout.size();
            self.allocated += 1 << target_order;

            // SAFETY: current_addr is a valid, aligned pointer into our data region.
            return Ok(unsafe { NonNull::new_unchecked(current_addr as *mut u8) });
        }

        Err(())
    }

    /// Deallocate a block previously returned by [`alloc`](Self::alloc).
    ///
    /// # Safety
    ///
    /// `ptr` and `layout` must exactly match a previous successful allocation
    /// from this specific `MetadataHeap` instance, and that allocation must
    /// not already have been deallocated.
    pub unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let current_addr: usize = ptr.as_ptr() as usize;
        let idx: usize = self.addr_to_idx(current_addr);

        // ---- Read stored order from metadata ------------------------------
        let stored_order: usize =
            unsafe { (*self.meta.add(idx)).order as usize };
        debug_assert_eq!(
            unsafe { (*self.meta.add(idx)).state },
            BlockState::Used,
            "dealloc on non-Used block head"
        );

        // Cross-check against the layout-derived order.
        let expected_order: usize = self.order_from_layout(layout);
        debug_assert_eq!(
            stored_order, expected_order,
            "stored order {} does not match layout-derived order {}",
            stored_order, expected_order,
        );

        let alloc_order: usize = stored_order; // save for stats

        // ---- 1. Clear allocated head to Reserved FIRST --------------------
        unsafe {
            (*self.meta.add(idx)).state = BlockState::Reserved;
        }

        // ---- 2. Merge loop ------------------------------------------------
        let mut current_addr: usize = current_addr;
        let mut current_order: usize = alloc_order;

        while current_order < ORDER - 1 {
            let buddy_addr: usize = current_addr ^ (1 << current_order);

            // Buddy out of data region → stop.
            if buddy_addr < self.heap_start || buddy_addr >= self.heap_end {
                break;
            }

            let buddy_idx: usize = self.addr_to_idx(buddy_addr);
            let buddy_meta: &BlockMeta =
                unsafe { &*self.meta.add(buddy_idx) };

            if buddy_meta.state != BlockState::Free {
                break;
            }
            if buddy_meta.order as usize != current_order {
                break;
            }

            // Unlink the buddy from its free-list.
            unsafe {
                self.unlink_free(buddy_idx);
            }

            current_addr = min(current_addr, buddy_addr);
            current_order += 1;
        }

        // ---- 3. Push the merged (or original) block -----------------------
        let final_idx: usize = self.addr_to_idx(current_addr);
        unsafe {
            self.push_free(final_idx, current_order);
        }

        // ---- 4. Update statistics -----------------------------------------
        self.user -= layout.size();
        self.allocated -= 1 << alloc_order; // original order, NOT merged
    }

    // ---- Statistics (matching Heap's API) ---------------------------------
    /// Total bytes requested by the user (sum of `layout.size()`).
    pub fn stats_alloc_user(&self) -> usize {
        self.user
    }

    /// Total bytes actually carved from the heap (power-of-two rounded).
    pub fn stats_alloc_actual(&self) -> usize {
        self.allocated
    }

    /// Total usable bytes in the data region.
    pub fn stats_total_bytes(&self) -> usize {
        self.total
    }

    /// Allocate a single block at the given order (page-granular allocation for slab).
    ///
    /// The order must satisfy `MIN_ORDER ≤ order.0 < ORDER`.
    pub fn alloc_pages(&mut self, order: PageOrder) -> Result<PageRun, AllocError> {
        let target_order = order.0 as usize;
        if target_order < MIN_ORDER || target_order >= ORDER {
            return Err(AllocError::InvalidOrder);
        }
        for i in target_order..ORDER {
            if self.free_area[i].len == 0 {
                continue;
            }
            let idx = self.pop_free(i).unwrap();
            let current_addr = self.idx_to_addr(idx);
            for j in (target_order + 1..=i).rev() {
                let buddy_addr = current_addr + (1 << (j - 1));
                let buddy_idx = self.addr_to_idx(buddy_addr);
                unsafe { self.push_free(buddy_idx, j - 1); }
            }
            unsafe {
                (*self.meta.add(idx)).state = BlockState::Used;
                (*self.meta.add(idx)).order = target_order as u8;
            }
            self.allocated += 1 << target_order;
            return Ok(PageRun {
                base: unsafe { NonNull::new_unchecked(current_addr as *mut u8) },
                order,
            });
        }
        Err(AllocError::NoMemory)
    }

    /// Deallocate a block previously returned by [`alloc_pages`](Self::alloc_pages).
    ///
    /// # Safety
    ///
    /// `run` must match a previous `alloc_pages` call and must not already be freed.
    pub unsafe fn dealloc_pages(&mut self, run: PageRun) {
        let current_addr = run.base.as_ptr() as usize;
        let idx = self.addr_to_idx(current_addr);
        let alloc_order = run.order.0 as usize;
        debug_assert_eq!(unsafe { (*self.meta.add(idx)).state }, BlockState::Used);
        unsafe { (*self.meta.add(idx)).state = BlockState::Reserved; }
        let mut current_addr = current_addr;
        let mut current_order = alloc_order;
        while current_order < ORDER - 1 {
            let buddy_addr = current_addr ^ (1 << current_order);
            if buddy_addr < self.heap_start || buddy_addr >= self.heap_end {
                break;
            }
            let buddy_idx = self.addr_to_idx(buddy_addr);
            let buddy_meta = unsafe { &*self.meta.add(buddy_idx) };
            if buddy_meta.state != BlockState::Free {
                break;
            }
            if buddy_meta.order as usize != current_order {
                break;
            }
            unsafe { self.unlink_free(buddy_idx); }
            current_addr = min(current_addr, buddy_addr);
            current_order += 1;
        }
        let final_idx = self.addr_to_idx(current_addr);
        unsafe { self.push_free(final_idx, current_order); }
        self.allocated -= 1 << alloc_order;
    }

    /// Return count of free blocks per order (histogram).
    pub fn free_block_counts(&self) -> [usize; ORDER] {
        let mut counts = [0usize; ORDER];
        for i in 0..ORDER {
            counts[i] = self.free_area[i].len;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl<const ORDER: usize, const MIN_ORDER: usize> MetadataHeap<ORDER, MIN_ORDER> {
    /// Map a data-region address → unit index (zero-based).
    #[inline]
    fn addr_to_idx(&self, addr: usize) -> usize {
        (addr - self.heap_start) >> MIN_ORDER
    }

    /// Map a unit index → data-region address.
    #[inline]
    fn idx_to_addr(&self, idx: usize) -> usize {
        self.heap_start + (idx << MIN_ORDER)
    }

    /// Compute the buddy-system order for a given layout.
    ///
    /// Rounds `size` up to the next power of two, enforces `MIN_ORDER`,
    /// and returns `Err(())` if the required order exceeds the maximum.
    fn layout_order(&self, layout: Layout) -> Result<usize, ()> {
        let size: usize = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let order: usize = size.trailing_zeros() as usize;
        if order >= ORDER {
            return Err(());
        }
        Ok(max(order, MIN_ORDER))
    }

    /// Like [`layout_order`] but returns the raw order (no error handling).
    /// Used for the debug_assert cross-check in [`dealloc`].
    fn order_from_layout(&self, layout: Layout) -> usize {
        let size: usize = max(
            layout.size().next_power_of_two(),
            max(layout.align(), size_of::<usize>()),
        );
        let order: usize = size.trailing_zeros() as usize;
        max(order, MIN_ORDER)
    }

    // ---- Free-list operations ---------------------------------------------

    /// Push the block starting at `idx` (order `order`) onto the free-list.
    ///
    /// # Safety
    ///
    /// - `idx` must be within `[0, meta_len)`.
    /// - The metadata for `idx` must currently be `Reserved`.
    /// - The data-region address must be aligned to `1 << order`.
    unsafe fn push_free(&mut self, idx: usize, order: usize) {
        debug_assert!(order < ORDER);
        debug_assert_eq!(
            unsafe { (*self.meta.add(idx)).state },
            BlockState::Reserved,
            "push_free on non-Reserved block"
        );
        debug_assert_eq!(
            self.idx_to_addr(idx) & ((1 << order) - 1),
            0,
            "push_free with misaligned address"
        );

        let old_head: usize = self.free_area[order].head;

        unsafe {
            *self.meta.add(idx) = BlockMeta {
                state: BlockState::Free,
                order: order as u8,
                prev: NIL,
                next: old_head,
            };
        }

        if old_head != NIL {
            unsafe {
                (*self.meta.add(old_head)).prev = idx;
            }
        }

        self.free_area[order].head = idx;
        self.free_area[order].len += 1;
    }

    /// Remove the block at `idx` from its free-list and reset it to `Reserved`.
    ///
    /// # Safety
    ///
    /// - `idx` must be within `[0, meta_len)`.
    /// - The metadata for `idx` must currently be `Free`.
    unsafe fn unlink_free(&mut self, idx: usize) {
        let meta_ref: &mut BlockMeta =
            unsafe { &mut *self.meta.add(idx) };
        debug_assert_eq!(meta_ref.state, BlockState::Free);

        let order: usize = meta_ref.order as usize;
        let prev: usize = meta_ref.prev;
        let next: usize = meta_ref.next;

        // Remove from doubly-linked list.
        if prev != NIL {
            unsafe {
                (*self.meta.add(prev)).next = next;
            }
        } else {
            self.free_area[order].head = next;
        }

        if next != NIL {
            unsafe {
                (*self.meta.add(next)).prev = prev;
            }
        }

        self.free_area[order].len -= 1;

        // Clear to Reserved.
        *meta_ref = BlockMeta::reserved();
    }

    /// Pop the head of the free-list for `order`.
    ///
    /// Returns `None` when the list is empty.
    fn pop_free(&mut self, order: usize) -> Option<usize> {
        if self.free_area[order].head == NIL {
            return None;
        }
        let idx: usize = self.free_area[order].head;
        // SAFETY: the head was placed by push_free and is Free.
        unsafe {
            self.unlink_free(idx);
        }
        Some(idx)
    }
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl<const ORDER: usize, const MIN_ORDER: usize> Default
    for MetadataHeap<ORDER, MIN_ORDER>
{
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<const ORDER: usize, const MIN_ORDER: usize> Send
    for MetadataHeap<ORDER, MIN_ORDER> {}

impl<const ORDER: usize, const MIN_ORDER: usize> fmt::Debug
    for MetadataHeap<ORDER, MIN_ORDER>
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MetadataHeap")
            .field("user", &self.user)
            .field("allocated", &self.allocated)
            .field("total", &self.total)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// LockedMetadataHeap
// ---------------------------------------------------------------------------

#[cfg(feature = "use_spin")]
pub struct LockedMetadataHeap<const ORDER: usize, const MIN_ORDER: usize>(
    Mutex<MetadataHeap<ORDER, MIN_ORDER>>,
);

#[cfg(feature = "use_spin")]
impl<const ORDER: usize, const MIN_ORDER: usize> LockedMetadataHeap<ORDER, MIN_ORDER> {
    pub const fn new() -> Self {
        LockedMetadataHeap(Mutex::new(MetadataHeap::new()))
    }
    pub const fn empty() -> Self {
        Self::new()
    }
}

#[cfg(feature = "use_spin")]
impl<const ORDER: usize, const MIN_ORDER: usize> Deref
    for LockedMetadataHeap<ORDER, MIN_ORDER>
{
    type Target = Mutex<MetadataHeap<ORDER, MIN_ORDER>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pages_alloc_dealloc_counts() {
        let mut heap = MetadataHeap::<8, 3>::empty();
        let mut buf = vec![0u8; 65536];
        let start = buf.as_mut_ptr() as usize;
        unsafe { heap.try_init(start, 65536).expect("init failed"); }

        let counts_before = heap.free_block_counts();

        let r1 = heap.alloc_pages(PageOrder(3)).expect("alloc order 3");
        let r2 = heap.alloc_pages(PageOrder(3)).expect("alloc order 3");
        let r3 = heap.alloc_pages(PageOrder(4)).expect("alloc order 4");
        let r4 = heap.alloc_pages(PageOrder(5)).expect("alloc order 5");

        // Sanity: free blocks should have changed (at least one order different).
        let counts_after = heap.free_block_counts();
        assert_ne!(counts_before, counts_after, "counts unchanged after alloc");

        unsafe { heap.dealloc_pages(r1); }
        unsafe { heap.dealloc_pages(r2); }
        unsafe { heap.dealloc_pages(r3); }
        unsafe { heap.dealloc_pages(r4); }

        let counts_restored = heap.free_block_counts();
        assert_eq!(counts_before, counts_restored, "counts not restored after dealloc");
    }

    #[test]
    fn test_alloc_pages_invalid_order() {
        let mut heap = MetadataHeap::<8, 3>::empty();
        let mut buf = vec![0u8; 65536];
        let start = buf.as_mut_ptr() as usize;
        unsafe { heap.try_init(start, 65536).expect("init failed"); }

        assert!(heap.alloc_pages(PageOrder(2)).is_err(), "order 2 should be too small");
        assert!(heap.alloc_pages(PageOrder(8)).is_err(), "order 8 should be too large");
    }

    #[test]
    fn test_alloc_pages_no_memory() {
        let mut heap = MetadataHeap::<8, 3>::empty();
        let mut buf = vec![0u8; 256];
        let start = buf.as_mut_ptr() as usize;
        if unsafe { heap.try_init(start, 256) }.is_ok() {
            let _ = heap.alloc_pages(PageOrder(7));
            let _ = heap.alloc_pages(PageOrder(7));
            assert!(heap.alloc_pages(PageOrder(7)).is_err(), "should be OOM");
        }
    }
}

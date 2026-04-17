# Triton Instrument Dialect and Concurrency Sanitizer (ConSan)

## Overview

ConSan instruments Triton IR with runtime checks for illegal concurrent access
to shared memory and tensor memory. The pass tracks a per-buffer frontier of
visible reads and writes, models mbarrier synchronization, and models
commit-count synchronization for asynchronous operations such as `cp.async`,
WGMMA, TMA store, and AMD TDM copies.

The pass is target-hook based. The module target selects the hook
implementation:

- `cuda:*` uses the NVIDIA hooks.
- `hip:*` uses the AMD hooks.

ConSan currently supports one public entry point in the module. It uses
BufferRegion analysis to collect shared-memory buffers, tensor-memory buffers,
and barrier allocations, then creates auxiliary state in distributed tensors and
shared-cluster global scratch memory. Most state has a leading CTA dimension so
cluster and multicast effects can be modeled explicitly.

## Thread Model

ConSan uses logical thread ids rather than hardware lane ids:

- Base threads: 16 logical warp-specialization slots. The default region is
  thread 0; warp-specialize partition regions use `partition_index + 1`.
- TMA peer threads: 16 additional slots at offset 16.
- Tensor Core peer threads: 16 additional slots at offset 32.
- Total logical slots in use: 48. Visibility masks are padded to 64 bits.

For a base thread, `getThreadPeersMask` returns the base thread plus its TMA and
Tensor Core peers. For a TMA or Tensor Core thread, it returns only that helper
thread. Commit-count tracking uses only the 16 base-thread columns, so helper
threads are folded back with `thread % 16` where commit counters are involved.

At a `ttg.warp_specialize`, the pass copies the default thread's read and write
visibility to the destination partition peer masks so partition-local execution
starts with the visibility frontier that existed before specialization.

## Auxiliary Data

All buffer and barrier counts are power-of-two padded. `C` is the number of
CTAs in the cluster, `B` is the number of tracked buffers for one memory type,
`K` is the number of tracked barriers, `T` is the logical ConSan thread bit
slots padded to 64, and `P` is the 16 base-thread commit columns.

The pass creates separate buffer, visibility, tracking, and alias state for
shared memory and tensor memory when that memory type is present.

- `buffers` (tensor, `<C x B x i64>`): Packed buffer descriptors. Each element
  stores a 32-bit base offset and 32-bit length.
- `barriers` (tensor, `<C x K x i64>`): Packed descriptors for mbarrier
  allocations. Barrier descriptors are shared-memory descriptors.
- `barrierStates` (scratch, `<C x K x i64>`): Packed barrier lifecycle state.
  Zero means invalid or uninitialized. Bit 0 is phase, bits `[1..20]` are the
  initial arrival count, bits `[21..40]` are the current arrival count, and
  bits `[41..61]` are a signed tx-count field.
- `barrierWriteRecipients` (scratch, `<C x K x i32>`): CTA bitsets recording
  write-recipient CTA rows reached by outstanding TMA-style barrier effects.
- `writeVisibility` (scratch, `<C x B x i64>`): Per-buffer bitmask. Bit `i`
  means logical thread `i` can see the latest write frontier for that buffer.
- `writeTracking` (scratch, `<C x B x K x i8>`): Buffer/barrier map tracking
  writes that a barrier can make visible.
- `readVisibility` (scratch, `<C x B x T x i64>`): Per-buffer, per-thread
  lanes. Each lane stores a bitmask of reads visible to that lane's thread.
- `readTracking` (scratch, `<C x B x K x i64>`): Buffer/barrier map tracking
  read visibility masks that a barrier can make visible.
- `commits[kind]` (scratch, `<C x B x P x i8>`): Per-commit-kind outstanding
  commit counters. Current kinds are `AsyncCp`, `Wgmma`, and `TmaStore`.
- `aliasMatrices` (tensor, `<C x B x B x i1>`): Optional per-memory-type alias
  matrix. It is only created when BufferRegion analysis finds cross-buffer
  aliasing. Runtime checks expand a selected buffer through this matrix.
- `lock` (scratch pointer, `i32`): A shared-cluster lock used to serialize
  instrumentation updates.
- `waiting` (scratch, `<C x K x i32>`): Deadlock-detection state. Each base
  thread has two bits: waiting flag and stored phase.

Scratch tensors are zero-initialized once by CTA 0, followed by a cluster
barrier for multi-CTA kernels or a global-memory barrier for single-CTA kernels.
The same setup initializes the ConSan lock.

## Memory Legality

For a read effect, ConSan checks that there is no outstanding write to the
selected buffer, or that the reading thread can see the latest write. For a
write effect, ConSan checks both write visibility and read visibility: the
writing thread must see the latest write frontier and all prior reads for the
selected buffer.

The runtime checks are:

- `verify_write_visibility`: Selects the buffer row, expands through aliases
  when needed, filters by recipient CTA rows, and succeeds if either no write is
  visible for that row or the current logical thread bit is present.
- `verify_read_visibility`: Selects the buffer row, expands aliases when
  needed, filters by recipient CTA rows, ORs all read-visibility lanes, and
  succeeds only if the current thread's lane is a superset of that total.
- `check_outstanding_commits`: For shared-memory accesses, checks the relevant
  commit-count table rows and asserts that no pending asynchronous access still
  protects the selected or aliased buffer.

After a barrier-tracked read, ConSan ORs the current peer thread mask into the
selected lanes of `readVisibility`. After a barrier-tracked write, ConSan
replaces `writeVisibility` for the selected buffer with the current peer mask,
then clears the buffer's write tracking, read visibility, and read tracking.

All normal instrumentation emitted around one IR operation is wrapped in the
ConSan lock. Barrier waits are split into a locked pre-wait section and a locked
post-wait section.

## CTA Recipients and Multicast

Most runtime helpers take a `recipientCTAs` bitset. This bitset is converted to
a tensor mask over the leading `C` dimension so only relevant CTA rows are
checked or updated.

The target hooks compute recipients from the operation:

- Non-multicast operations usually target the current CTA.
- Multicast TMA loads update all result-recipient CTAs, while barrier arrivals
  route to the leader barrier CTA.
- NVIDIA two-CTA Tensor Core operations are predicated to the issuing CTA pair
  leader.
- TMA load effects that write one set of CTAs and signal a different leader
  barrier use `barrierWriteRecipients` so a later wait can transfer write
  visibility to both the waiting CTA and the effect-recipient CTA rows.

## Barrier Synchronization

ConSan separates barrier tracking from visibility transfer.

For frontier-tracked barriers, an arrive or commit snapshots the current
thread's visible writes and reads into the barrier's tracking rows:

- `track_visible_writes` records buffer rows whose write frontier is visible to
  the arriving thread.
- `track_visible_reads` records the arriving thread's visible read mask for
  each buffer.

Some effects use precise write tracking instead of frontier tracking. For
example, NVIDIA TMA loads use `track_barrier_write_for_buffer` to mark only the
buffers written by that operation and to remember effect-recipient CTA rows.

On a barrier wait, ConSan:

1. Acquires the ConSan lock.
2. Verifies the barrier is initialized.
3. Sets the current base thread's waiting flag and phase.
4. Checks whether all active base threads are waiting on matching barrier
   phases.
5. Releases the lock and lets the real wait execute.
6. Re-acquires the lock after the wait.
7. Transfers tracked write and read visibility from the barrier to the current
   thread's peer mask for shared memory and tensor memory.
8. Clears the current base thread's waiting bits.

Write transfers also consult `barrierWriteRecipients`, which lets TMA-style
cross-CTA writes become visible in the CTA rows reached by the memory effect.
Read transfers update the current CTA row.

## Barrier Lifecycle and Deadlock Checks

The barrier state table models initialized, invalidated, phase, arrival-count,
and tx-count behavior:

- `verify_barrier_can_init` asserts that the selected barrier state is zero
  before initialization.
- `init_barrier_state` sets phase 0 and stores the initial/current arrival
  count. A zero state remains the invalid/uninitialized sentinel.
- `verify_barrier_initialized` asserts that a barrier is initialized before it
  is used.
- `verify_barrier_arrive` checks that subtracting the arrive count will not
  underflow the current count and that adding the tx-count delta remains within
  the signed tx-count field range.
- `update_barrier_state` subtracts the arrive count, adds the tx-count delta,
  flips phase when both current arrival count and tx-count reach zero, reloads
  the current count from the initial count, and clears tx-count for the new
  phase.
- `invalidate_barrier_state` clears the barrier state and waiting bits.
  Barrier invalidation also clears barrier read/write tracking for both memory
  types.

Deadlock detection uses `waiting`. The check aligns the stored per-thread
waiting phase with each barrier's current phase, filters to active base threads,
and asserts if every active thread is waiting on a matching phase.

## Commit-Count Synchronization

Commit-count synchronization is used for operations whose completion is ordered
by outstanding commit groups rather than by a barrier. It is only modeled for
shared-memory buffers.

Each commit table entry is:

- `0`: no outstanding access.
- `-1`: access staged but not yet committed.
- Positive value: committed access with an outstanding-group distance.

The helpers are:

- `stage_access_for_commit`: Marks matching buffer rows in the current
  base-thread column as `-1`.
- `commit_accesses`: Converts `-1` to `1` and increments positive entries in
  the committing base-thread column.
- `clear_outstanding_commits_transfer_writes`: Clears entries with value
  greater than the wait's pending-count threshold and transfers write
  visibility to the provided peer mask.
- `clear_outstanding_commits_transfer_reads`: Same, but transfers read
  visibility.
- `clear_outstanding_commits_transfer_both`: Clears once and transfers both
  read and write visibility when both visibility tables are present.

Before shared-memory reads and writes, ConSan checks target-defined outstanding
commit kinds. The check inspects all relevant CTA/buffer rows, expands aliases
when necessary, and can exclude the caller's own base-thread column for ordered
commit kinds. That exclusion is used by targets whose operations complete in
issue order within one ConSan logical partition, avoiding false positives for
same-partition ordering while still checking cross-partition races.

## Target Coverage

The common hook implementation covers these TritonGPU operations:

- `ttg.async_copy_global_to_local`: shared-memory write tracked with
  `AsyncCp` commit counts.
- `ttg.async_commit_group`: commits staged `AsyncCp` accesses.
- `ttg.async_wait`: clears `AsyncCp` entries beyond the pending-count threshold
  and transfers write visibility.
- `ttg.local_load`: barrier-tracked shared-memory read.
- `ttg.local_store`: barrier-tracked shared-memory write.
- `ttg.local_alloc` with a source: barrier-tracked shared-memory write.

NVIDIA hooks additionally cover:

- `ttng.init_barrier`, `ttng.wait_barrier`, and `ttng.inval_barrier` lifecycle
  and wait instrumentation.
- `ttng.barrier_expect`, including tx-count accounting and the non-leader CTA
  arrive path for multicast barriers.
- `ttng.arrive_barrier`.
- TMA loads as barrier-tracked writes with tx-count decrement and precise
  effect-write tracking.
- TMA stores as `TmaStore` commit-count reads, with `ttng.tma_store_wait`
  transferring read visibility.
- TMEM load, store, alloc-with-source, and copy operations.
- TCGen5 MMA, scaled MMA, commit, and TMEM copy operations as Tensor Core peer
  thread effects.
- Async WGMMA operands in shared memory as `Wgmma` commit-count reads, with
  `ttng.warp_group_dot_wait` transferring read visibility.

AMD hooks additionally cover:

- AMD barrier init and wait instrumentation.
- AMD explicit barrier arrives, with arrive count scaled by warps and threads
  per warp.
- Async TDM global-to-local and local-to-global copies. With a barrier, these
  are modeled through barrier arrivals; without a barrier, they use `TmaStore`
  commit counts and implicit commits.
- AMD async wait variants for `AsyncCp`, and TDM wait variants for `TmaStore`
  commit counts.
- Ordered TDM commit kinds, using the self-column exclusion described above.

## Current Limitations

- ConSan models one logical thread per warp-specialization partition, not every
  hardware lane. Target hooks compensate for known ordered same-partition
  commit flows where possible.
- Commit-count synchronization is only implemented for shared-memory buffers.
- Global-memory race checking is handled by the separate global sanitizer, not
  by ConSan.
- The pass expects exactly one public entry point in the module.

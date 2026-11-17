//! qmmf's cluster fit: a split meets in a cluster only while every cluster of its grid is resident (#3681).
const std = @import("std");
const q = @import("qmmf.zig");
const launch = @import("launch.zig");

/// Resident clusters of each cluster instance (0: unknown): [mode][out][16-, 32-, 64-row tile, half tile][sk 2, 4, 8].
pub const Fit = [q.modes.len][2][4][3]u16;

/// A call's launch as the lane picks it: the slice plan, half or full columns, and the clusters its grid holds.
pub const Shape = struct { plan: q.Plan, half: bool, tile: usize, row_tiles: usize, cols: usize };

fn sizeIndex(sk: u32) ?usize {
    return switch (sk) {
        2 => 0,
        4 => 1,
        8 => 2,
        else => null,
    };
}

/// The driver's resident clusters for each cluster instance and size; 0 where it cannot say.
pub fn measureFit(l: *const q.Lane) Fit {
    var fit: Fit = std.mem.zeroes(Fit);
    for (q.modes, 0..) |mode, mi| for (0..2) |oi| for (0..4) |ti| for ([_]u32{ 2, 4, 8 }, 0..) |sk, si| {
        const f = if (ti == 3) (if (q.halfModeIndex(mode)) |hm| l.half[hm][oi][1] else continue) else (if (oi == 1) l.f32_fns else l.fns)[mi][ti * 2 + 1];
        const shared = if (ti == 3) q.halfShared(mode) else q.sharedBytes(mode, q.row_tiles[ti]);
        const cfg: launch.Config = .{ .grid = .{ .x = 1, .y = 1, .z = sk }, .block = .{ .x = q.threads }, .shared = shared, .cluster = .{ .x = 1, .y = 1, .z = sk } };
        fit[mi][oi][ti][si] = @intCast(@min(launch.activeClusters(f, cfg) orelse 0, std.math.maxInt(u16)));
    };
    return fit;
}

/// The launch of `m` rows by `w`: a cluster only while its clusters fit (both outputs' smaller fit), else the reduce.
pub fn shape(l: q.Lane, m: usize, w: q.Weight) Shape {
    var p = q.planFor(m, w, l.clusters);
    const tiles = (w.n + q.bn - 1) / q.bn;
    const rows_t = (m + p.bm - 1) / p.bm;
    const half = q.halfModeIndex(w.mode) != null and p.bm == 16 and !p.fused and rows_t * tiles * p.sk < l.half_below;
    const cols: usize = if (half) 32 else q.bn;
    const ti: usize = if (half) 3 else if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
    if (p.cluster) if (sizeIndex(p.sk)) |si| {
        const a = l.fit[q.modeIndex(w.mode)][0][ti][si];
        const b = l.fit[q.modeIndex(w.mode)][1][ti][si];
        const clusters = rows_t * ((w.n + cols - 1) / cols);
        if (a != 0 and b != 0 and clusters > @min(a, b)) {
            p.cluster = false;
            p.reduce = true;
        }
    };
    return .{ .plan = p, .half = half, .tile = ti, .row_tiles = rows_t, .cols = cols };
}

/// The most partials any call of 1 .. `rows` rows needs (a split's need is not monotonic in rows): size scratch so.
pub fn maxPartBytes(l: q.Lane, w: q.Weight, rows: usize) usize {
    var most: usize = 0;
    for (1..@min(rows, q.fused_rows - 1) + 1) |m| most = @max(most, l.partBytes(m, w));
    return most;
}

/// A lane with no device for host tests: no kernels loaded, today's plan (fit unknown), `clusters` as given.
pub fn mock(clusters: bool) q.Lane {
    var l: q.Lane = undefined;
    l.clusters = clusters;
    l.fit = std.mem.zeroes(Fit);
    l.half_below = q.half_below_default;
    l.pdl = false;
    return l;
}

test "a split meets in a cluster only while its grid's clusters fit, from the smaller fit of both outputs" {
    var lane = mock(true);
    const fp8g = q.modeIndex(.fp8g);
    const w8 = try q.pinned(q.Weight{ .mode = .fp8g, .codes = 0, .scales = 0, .n = 2048, .k = 4096, .npad = 2048 }, 8);
    // unknown fit: today's plan, a cluster of 8 and no partials
    try std.testing.expect(lane.shape(1, w8).plan.cluster);
    try std.testing.expectEqual(@as(usize, 0), lane.partBytes(1, w8));
    // GB10's 12 clusters of 8 hold 32 of them in three waves: plain CTAs and the reduce
    lane.fit[fp8g][0][0][2] = 12;
    lane.fit[fp8g][1][0][2] = 12;
    const s = lane.shape(1, w8);
    try std.testing.expect(!s.plan.cluster and s.plan.reduce and !s.half);
    try std.testing.expectEqual(@as(usize, 8 * 1 * 2048 * 4), lane.partBytes(1, w8));
    lane.fit[fp8g][1][0][2] = 0; // one output unknown: the cluster stays
    try std.testing.expect(lane.shape(1, w8).plan.cluster);
    lane.fit[fp8g][1][0][2] = 48;
    try std.testing.expect(lane.shape(1, w8).plan.reduce); // the smaller fit decides
    const w4 = try q.pinned(w8, 4);
    lane.fit[fp8g][0][0][1] = 24;
    lane.fit[fp8g][1][0][1] = 24;
    try std.testing.expect(lane.shape(1, w4).plan.reduce);
    lane.fit[fp8g][0][0][1] = 48;
    lane.fit[fp8g][1][0][1] = 48;
    try std.testing.expect(lane.shape(1, w4).plan.cluster);
    // a half tile's grid has twice the columns: 1-16 rows reduce, 17 rows (full 32-row tile) keep the cluster
    const mx = q.modeIndex(.mxfp8);
    const narrow = try q.pinned(q.Weight{ .mode = .mxfp8, .codes = 0, .scales = 0, .n = 512, .k = 4096, .npad = 512 }, 8);
    lane.fit[mx][0][3][2] = 12;
    lane.fit[mx][1][3][2] = 12;
    lane.fit[mx][0][1][2] = 12;
    lane.fit[mx][1][1][2] = 12;
    try std.testing.expect(lane.shape(1, narrow).half and lane.shape(1, narrow).plan.reduce);
    try std.testing.expect(!lane.shape(17, narrow).half and lane.shape(17, narrow).plan.cluster);
    try std.testing.expectEqual(@as(usize, 0), lane.partBytes(17, narrow));
    try std.testing.expectEqual(@as(usize, 0), lane.partBytes(64, narrow));
    try std.testing.expectEqual(@as(usize, 8 * 16 * 512 * 4), lane.maxPartBytes(narrow, 64));
    try std.testing.expectEqual(@as(usize, 0), lane.maxPartBytes(narrow, 0));
    // fused rows never reduce
    try std.testing.expectEqual(@as(usize, 0), lane.partBytes(q.fused_rows, w8));
}

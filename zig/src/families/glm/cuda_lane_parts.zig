//! The lane slice partials a GLM forward needs at up to `rows` rows: each FP8, MXFP8 and NVFP4 lane weight's most.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const projection = @import("cuda_proj.zig");

fn proj(lane: *const cuda.qmmf.Lane, p: projection.Proj, rows: usize) usize {
    return switch (p) {
        .bf16 => 0, // the glue's matmuls size their own (b16.partBytes)
        .fp8g, .mxfp8 => |q| lane.maxPartBytes(q, rows),
    };
}

fn layer(lane: *const cuda.qmmf.Lane, l: weights.Layer, rows: usize) usize {
    var most: usize = 0;
    switch (l.mixer) {
        .kda => |k| {
            if (k.proj == .split) most = lane.maxPartBytes(k.proj.split.qkv, rows);
            most = @max(most, proj(lane, k.o, rows));
        },
        .mla => |a| for ([_]projection.Proj{ a.proj, a.q_b, a.o, a.index.kw, a.index.qb }) |p| {
            most = @max(most, proj(lane, p, rows));
        },
    }
    switch (l.ffn) {
        .dense => |m| switch (m.w) {
            .nvfp4 => |x| for ([_]cuda.qmmf.Weight{ x.gate, x.up, x.down }) |q| {
                most = @max(most, lane.maxPartBytes(q, rows));
            },
            .mxfp8 => |x| for ([_]cuda.qmmf.Weight{ x.gu, x.down }) |q| {
                most = @max(most, lane.maxPartBytes(q, rows));
            },
            .bf16 => {},
        },
        .moe => |m| most = @max(most, @max(proj(lane, m.shared_gu, rows), proj(lane, m.shared_down, rows))),
    }
    return most;
}

/// Bytes of fp32 partials the forward's lane splits need at up to `rows` rows; 0 without a lane.
pub fn bytes(lane_: ?*const cuda.qmmf.Lane, w: *const weights.Weights, rows: usize) usize {
    const lane = lane_ orelse return 0;
    var most = proj(lane, w.head, rows);
    for (w.layers) |l| most = @max(most, layer(lane, l, rows));
    if (w.mtp) |m| most = @max(most, layer(lane, m.layer, rows));
    return most;
}

test "the forward's partials cover a split that meets through the reduce, and none without a lane" {
    const lane = cuda.qmmf.Lane.mock(false); // every split reduces
    const head: projection.Proj = .{ .fp8g = cuda.fp8.weight(0, 0, 2048, 4096) };
    const w: weights.Weights = .{ .gpa = undefined, .config = undefined, .share = undefined, .head = head };
    try std.testing.expectEqual(@as(usize, 0), bytes(null, &w, 64));
    try std.testing.expectEqual(lane.maxPartBytes(head.fp8g, 64), bytes(&lane, &w, 64));
    try std.testing.expect(bytes(&lane, &w, 64) > 0);
    try std.testing.expectEqual(bytes(&lane, &w, 255), bytes(&lane, &w, 4096)); // prompt rows past the reduce add none
}

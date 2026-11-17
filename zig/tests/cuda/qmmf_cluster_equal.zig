//! tf-cuda-test qmmf-cluster-equal: a split met in a cluster and the same split met through the reduce, byte-equal.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const q = cuda.qmmf;
const k: usize = 4096; // 64 groups: K slices 2, 4 and 8 all pin
const row_counts = [_]usize{ 1, 16, 17, 32, 33, 64 }; // each row tile and its edge
const widths = [_]usize{ 512, 2048 }; // half tiles at 512 (MXFP8, BF16), full tiles at 2048 from 4 slices

/// Random finite codes and scales in `mode`'s packed sizes (any layout is a weight for an equality check).
fn weight(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, mode: q.Mode, n: usize) !q.Weight {
    const npad = (n + 127) / 128 * 128;
    const code_bytes: usize = switch (mode) {
        .fp4 => npad * k / 2,
        .fp8, .mxfp8, .fp8g => npad * k,
        .bf16 => npad * k * 2,
    };
    const codes = try a.alloc(u8, code_bytes);
    if (mode == .bf16) {
        const v = r.bfs(npad * k, 0.05);
        @memcpy(codes, std.mem.sliceAsBytes(v));
    } else rng.bytes(codes);
    if (mode != .fp4 and mode != .bf16) for (codes) |*c| {
        c.* &= 0xF7; // no e4m3 NaN (0x7F, 0xFF)
    };
    const scales = try a.alloc(u8, npad * k / 16);
    switch (mode) {
        .fp4 => for (scales) |*s| {
            s.* = 0x30 + rng.uintLessThan(u8, 16); // e4m3 near 1
        },
        .mxfp8 => for (scales) |*s| {
            s.* = 124 + rng.uintLessThan(u8, 8); // e8m0 near 1
        },
        .fp8g => for (std.mem.bytesAsSlice(f32, scales)) |*s| {
            s.* = 0.5 + rng.float(f32);
        },
        .fp8, .bf16 => {},
    }
    const sp = if (mode == .fp8 or mode == .bf16) 0 else try r.dev(scales);
    return .{ .mode = mode, .codes = try r.dev(codes), .scales = sp, .scale = 1, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
}

/// The lane's launches, recorded and forwarded to the real driver: kernels with a cluster, without, and reduces.
const Seen = struct {
    var real_ex: @FieldType(cuda.abi.Api, "cuLaunchKernelEx") = undefined;
    var real: @FieldType(cuda.abi.Api, "cuLaunchKernel") = undefined;
    var reduces: [2]cuda.abi.Function = .{ null, null };
    var clustered: usize = 0;
    var plain: usize = 0;
    var reduced: usize = 0;

    fn ex(lc: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, a: ?[*]?*anyopaque, e: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        var cl = false;
        for (lc.attrs.?[0..lc.num_attrs]) |at| cl = cl or at.id == .cluster_dimension;
        if (cl) clustered += 1 else plain += 1;
        return real_ex(lc, f, a, e);
    }
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, st: cuda.abi.Stream, a: ?[*]?*anyopaque, e: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        if (f == reduces[0] or f == reduces[1]) reduced += 1 else plain += 1;
        return real(f, gx, gy, gz, bx, by, bz, sm, st, a, e);
    }
    fn reset() void {
        clustered = 0;
        plain = 0;
        reduced = 0;
    }
};

/// Every output element finite, so equal bytes are not two equal sentinels.
fn finite(comptime T: type, bytes: []const u8) bool {
    if (T == u16) {
        for (std.mem.bytesAsSlice(u16, bytes)) |h| if (h & 0x7f80 == 0x7f80) return false;
    } else for (std.mem.bytesAsSlice(f32, bytes)) |v| if (!std.math.isFinite(v)) return false;
    return true;
}

pub fn run(gpu: check.Gpu) !void {
    if (try gpu.ctx.capability() < 90) return check.pass("qmmf-cluster-equal: no clusters below sm_90; skipped", .{});
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x3681);
    var r = try rig.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var d = gpu.d.*; // the lane's launches go through this copy's recording entries
    Seen.real_ex = d.api.cuLaunchKernelEx;
    Seen.real = d.api.cuLaunchKernel;
    d.api.cuLaunchKernelEx = Seen.ex;
    d.api.cuLaunchKernel = Seen.launch;
    var lane = try q.Lane.load(&d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    Seen.reduces = .{ lane.reduce[0].handle, lane.reduce[1].handle };
    var cases: usize = 0;
    for (q.modes) |mode| for (widths) |n| {
        const base = try weight(&r, a, prng.random(), mode, n);
        for ([_]u32{ 2, 4, 8 }) |sk| for (row_counts) |m| {
            const w = try q.pinned(base, sk);
            const x = try r.dev(std.mem.sliceAsBytes(r.bfs(m * k, 1.0)));
            const part = try r.out(@max(16, @as(usize, sk) * m * n * 4));
            inline for (.{ q.Out.bf16, q.Out.f32 }) |out| {
                const bytes = m * n * (if (out == .f32) @as(usize, 4) else 2);
                var got: [2][]u8 = undefined;
                for ([_]bool{ false, true }, 0..) |cl, i| {
                    lane.clusters = cl;
                    lane.fit = if (cl) @splat(@splat(@splat(@splat(std.math.maxInt(u16))))) else std.mem.zeroes(q.Fit);
                    const sh = lane.shape(m, w);
                    try check.expect(sh.plan.cluster == cl and sh.plan.reduce == !cl, "qmmf-cluster-equal: {t} n {d} sk {d} m {d}: cluster {}, reduce {}", .{ mode, n, sk, m, sh.plan.cluster, sh.plan.reduce });
                    const y = try r.out(bytes);
                    Seen.reset();
                    try lane.matmulOut(r.s, x, k, m, w, y, out, part);
                    const ran = if (cl) Seen.clustered == 1 and Seen.plain == 0 and Seen.reduced == 0 else Seen.clustered == 0 and Seen.plain == 1 and Seen.reduced == 1;
                    try check.expect(ran, "qmmf-cluster-equal: {t} n {d} sk {d} m {d} cluster {}: launches {d} clustered, {d} plain, {d} reduces", .{ mode, n, sk, m, cl, Seen.clustered, Seen.plain, Seen.reduced });
                    got[i] = try r.back(u8, y, bytes);
                    try check.expect(finite(if (out == .f32) f32 else u16, got[i]), "qmmf-cluster-equal: {t} n {d} sk {d} m {d}: a non-finite output", .{ mode, n, sk, m });
                }
                try check.sameBytes("qmmf-cluster-equal: reduce against cluster", got[0], got[1]);
                cases += 1;
            }
        };
    };
    try control(&r, &lane);
    check.pass("qmmf-cluster-equal: {d} cases (5 modes, both outputs, rows 1-64, sk 2/4/8, half and full tiles) byte-equal; the reversed reduce goes red", .{cases});
}

/// The reduce sums slices in order: 1e8, 1, -1e8, 1 gives 1; the same slices last first give 0, which must differ.
fn control(r: *rig.Rig, lane: *const q.Lane) !void {
    var got: [2]f32 = undefined;
    for ([2][4]f32{ .{ 1e8, 1, -1e8, 1 }, .{ 1, -1e8, 1, 1e8 } }, 0..) |slices, i| {
        const part = try r.dev(std.mem.sliceAsBytes(&slices));
        const y = try r.out(4);
        var args: cuda.launch.Args = .{};
        args.add(part);
        args.add(y);
        args.add(@as(i64, 1));
        args.add(@as(i32, 4));
        args.add(@as(f32, 1));
        try cuda.launch.launch(lane.reduce[@backingInt(q.Out.f32)], .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, r.s, &args);
        got[i] = (try r.back(f32, y, 1))[0];
    }
    try check.expect(got[0] == 1 and got[1] == 0, "qmmf-cluster-equal control: forward {d}, reversed {d} (want 1 and 0)", .{ got[0], got[1] });
    check.pass("qmmf-cluster-equal control: the slices in reverse order sum to {d}, not {d}: red as planted", .{ got[1], got[0] });
}

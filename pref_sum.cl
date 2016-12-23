
size_t calculate_wg_part(size_t n) {
    size_t work_groups = get_global_size(0);
    size_t m = max(work_groups, (size_t)(1 << (size_t)ceil(log2((float)n))));
    return m / work_groups;
}

void input_to_local(global float const* input, size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t wg_part = calculate_wg_part(n);

    size_t i = wg_part * id;
    for ( ; i < wg_part * (id + 1) && i < n; ++i) {
        loc_mem[i % wg_part] = input[i];
    }
    for ( ; i < wg_part * (id + 1); ++i) {
        loc_mem[i % wg_part] = 0.0f;
    }
}

void reduce(size_t wg_part, local float* loc_mem) {
    size_t offset = 0;
    while (wg_part > 1) {
        size_t prev_offset = offset;
        offset += wg_part;
        wg_part /= 2;

        for (size_t i = 0; i < wg_part; ++i) {
            loc_mem[offset + i] = loc_mem[prev_offset + 2 * i] + loc_mem[prev_offset + 2 * i + 1];
        }
    }
}

void last_row_to_output(size_t wg_part, local float* loc_mem, global float* output) {
    size_t id = get_global_id(0);
    output[id] = loc_mem[2 * wg_part - 2];
}

void last_row_to_local(size_t wg_part, local float* loc_mem, global float const* input) {
    size_t id = get_global_id(0);
    loc_mem[2 * wg_part - 2] = input[id];;
}

void downsweep(size_t wg_part, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t chunk = 1;
    size_t offset = 2 * wg_part - 2;

    while (chunk != wg_part) {
        size_t prev_offset = offset;
        offset -= 2 * chunk;

        for (size_t i = 0; i < chunk; ++i) {
            loc_mem[offset + 2 * i + 1] = loc_mem[prev_offset + i] + loc_mem[offset + 2 * i];
            loc_mem[offset + 2 * i] = loc_mem[prev_offset + i];
        }
        chunk *= 2;
    }
}

void local_to_output(size_t n, local float* loc_mem, global float* output) {
    size_t id = get_global_id(0);
    size_t wg_part = calculate_wg_part(n);

    for (size_t i = wg_part * id; i < wg_part * (id + 1) && i < n; ++i) {
        output[i] = loc_mem[i % wg_part];
    }
}

void kernel first_stage(global float const* input, size_t n,
                        local float* loc_mem, global float* outp_last_row) {
    input_to_local(input, n, loc_mem);
    size_t wg_part = calculate_wg_part(n);
    reduce(wg_part, loc_mem);
    last_row_to_output(wg_part, loc_mem, outp_last_row);
}

void kernel second_stage(global float const* input, global float const* inp_last_row,
                         size_t n, local float* loc_mem, global float* output) {
    input_to_local(input, n, loc_mem);
    size_t wg_part = calculate_wg_part(n);
    reduce(wg_part, loc_mem);
    last_row_to_local(wg_part, loc_mem, inp_last_row);
    downsweep(wg_part, loc_mem);
    local_to_output(n, loc_mem, output);
}

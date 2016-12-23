void input_to_local(global float const* input, size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);

    size_t m = 1 << (size_t)ceil(log2((float)n));
    size_t wi_part = m / work_items;

    size_t i = wi_part * id;
    for ( ; i < wi_part * (id + 1) && i < n; ++i) {
        loc_mem[i] = input[i];
    }
    for ( ; i < wi_part * (id + 1); ++i) {
        loc_mem[i] = 0.0f;
    }
}

void to_root(size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);
    size_t log_work_items = log2((float)work_items);

    size_t m = ceil(log2((float)n));
    size_t chunk = 1 << m;
    size_t offset = 0;
    size_t wi_part = chunk / work_items;

    for (size_t d = m - 1; d >= log_work_items; --d) {
        size_t prev_offset = offset;
        offset += chunk;
        chunk /= 2;
        wi_part /= 2;

        for (size_t i = wi_part * id; i < wi_part * (id + 1); ++i) {
            loc_mem[offset + i] = loc_mem[prev_offset + 2 * i] + loc_mem[prev_offset + 2 * i + 1];
        }
    }
}

void sum_last_row(size_t n, local float* loc_mem) {
    if (get_global_id(0) != 0)
        return;

    size_t work_items = get_global_size(0);
    size_t m = ceil(log2((float)n));
    size_t offset = (1 << (m + 1)) - 2 * work_items;

    for (size_t i = 1; i < work_items; ++i)
        loc_mem[offset + i] += loc_mem[offset + i - 1];
    for (size_t i = 1; i < work_items; ++i)
        loc_mem[offset + i] = loc_mem[offset + i - 1];
    loc_mem[offset] = 0;
}

void from_root(size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);
    size_t log_work_items = log2((float)work_items);

    size_t m = ceil(log2((float)n));
    size_t chunk = work_items;
    size_t offset = (1 << (m + 1)) - 2 * work_items;
    size_t wi_part = 1;

    for (size_t d = log_work_items; d < m; ++d) {
        size_t prev_offset = offset;
        chunk *= 2;
        offset -= chunk;
        wi_part *= 2;

        for (size_t i = wi_part * id; i < wi_part * (id + 1); ++i) {
            loc_mem[offset + 2 * i + 1] = loc_mem[prev_offset + i] + loc_mem[offset + 2 * i];
            loc_mem[offset + 2 * i] = loc_mem[prev_offset + i];
        }
    }
}

void local_to_output(size_t n, local float* loc_mem, global float* output) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);

    size_t wi_part = ceil(n * 1.0f / work_items);
    size_t i;
    for (i = wi_part * id; i < wi_part * (id + 1) && i < n; ++i) {
        output[i] = loc_mem[i];
    }
}

void temp_output(size_t n, local float* loc_mem, global float* output) {
    if (get_global_id(0) != 0)
        return;
    for (size_t i = 0; i < n; ++i)
        output[i] = loc_mem[i];
}

void kernel prefix_sum(global float const* input, size_t n,
                       local float* loc_mem, global float* output) {
    printf("%d %d ; %d %d\n", get_global_size(0), get_global_id(0), get_local_size(0), get_local_id(0));

    input_to_local(input, n, loc_mem);

    barrier(CLK_LOCAL_MEM_FENCE);
    local_to_output(n, loc_mem, output);
    barrier(CLK_LOCAL_MEM_FENCE);
    return;

    to_root(n, loc_mem);

    printf(">>>>>>>");
    barrier(CLK_LOCAL_MEM_FENCE);
    printf("<<<<<<<");
    sum_last_row(n, loc_mem);
    barrier(CLK_LOCAL_MEM_FENCE);

    from_root(n, loc_mem);
    local_to_output(n, loc_mem, output);
}

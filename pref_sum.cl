void input_to_local(global float const* input, size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);

    size_t wi_part = ceil(n * 1.0f / work_items);
    size_t i;
    for (i = wi_part * id; i < wi_part * (id + 1) && i < n; ++i) {
        loc_mem[i] = input[i];
    }
    for ( ; i < wi_part * (id + 1); ++i) {
        loc_mem[i] = 0.0f;
    }
}

void to_root(global float const* input, size_t n, local float* loc_mem) {
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

    if (id == 0) {
        for (size_t i = 1; i < work_items; ++i)
            loc_mem[offset + i] += loc_mem[offset + i - 1];
        for (size_t i = 1; i < work_items; ++i)
            loc_mem[offset + i] = loc_mem[offset + i - 1];
        loc_mem[offset] = 0;
    }
}

void from_root(global float const* input, size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size(0);
    size_t log_work_items = log2((float)work_items);

    size_t m = ceil(log2((float)n));
    size_t chunk = work_items;
    size_t offset = (1 << (m + 1)) - 2 * work_items - 1;
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

void kernel prefix_sum(global float const* input, size_t n,
                       local float* loc_mem, global float* output) {
    // printf("Prefix sum\n");
    // printf("Work items: %d\n", get_global_size(0));
    // printf("Work id:    %d\n", get_global_id(0));
    //
    // printf("Input to local\n");
    input_to_local(input, n, loc_mem);
    // printf("To root\n");
    to_root(input, n, loc_mem);
    // printf("Barrier\n");
    barrier(CLK_LOCAL_MEM_FENCE);
    // printf("From root\n");
    from_root(input, n, loc_mem);
    // printf("Local to output\n");
    local_to_output(n, loc_mem, output);
    // printf("Success\n");
}

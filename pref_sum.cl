#include <cmath>
#include <iostream>

size_t get_work_items() {
    static size_t work_items
}

void input_to_local(global float const* input, size_t n, local float* loc_mem) {
    size_t id = get_global_id(0);
    size_t work_items = get_global_size();

    size_t wi_part = std::ceil(n * 1.0f / work_items);
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
    size_t work_items = get_global_size();
    size_t log_work_items = static_cast<size_t>(std::log2(work_items))

    size_t m = std::ceil(std::log2(n));
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
    size_t work_items = get_global_size();
    size_t log_work_items = static_cast<size_t>(std::log2(work_items))

    size_t m = std::ceil(std::log2(n));
    size_t chunk = work_items;
    size_t offset = (1 << m + 1) - 2 * work_items - 1;
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
    size_t work_items = get_global_size();

    size_t wi_part = std::ceil(n * 1.0f / work_items);
    size_t i;
    for (i = wi_part * id; i < wi_part * (id + 1) && i < n; ++i) {
        output[i] = loc_mem[i];
    }
}

void kernel prefix_sum(global float const* input, size_t n,
                       local float* loc_mem, global float* output) {
    std::cout << "Prefix sum" << std::endl;
    std::cout << "Work items: " << get_global_size() << std::endl;
    std::cout << "Work id: " << get_global_id() << std::endl;

    std::cout << "Input to local" << std::endl;
    input_to_local(input, n, loc_mem);
    std::cout << "To root" << std::endl;
    to_root(input, n, loc_mem);
    std::cout << "Barrier" << std::endl;
    barrier(CLK_LOCAL_MEM_FENCE);
    std::cout << "From root" << std::endl;
    from_root(input, n, loc_mem);
    std::cout << "Local to output" << std::endl;
    local_to_output(n, loc_mem, output);
    std::cout << "Success" << std::endl;
}

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include <CL/cl.hpp>

const size_t work_groups = 256;
using data_t = int;

void initialize(cl::Platform& default_platform, cl::Device& default_device) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (!all_platforms.size()) {
        std::cerr << "No platforms found." << std::endl;
        exit(1);
    }

    default_platform = all_platforms[0];
    std::cerr << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (!all_devices.size()) {
        std::cerr << "No devices found." << std::endl;
        exit(1);
    }

    default_device = all_devices[0];
    std::cerr << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cerr << "Memory limit: " << default_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
}

std::string load_kernel(std::string const& filename) {
    std::ifstream kernel_file(filename);
    std::string kernel_code;
    while (kernel_file) {
        std::string tmp;
        getline(kernel_file, tmp);
        kernel_code += tmp + "\n";
    }
    return kernel_code;
}

void sum_last_row(data_t* last_row, size_t size) {
    for (size_t i = 1; i < size; ++i)
        last_row[i] += last_row[i - 1];
    for (size_t i = size; i; --i)
        last_row[i] = last_row[i - 1];
    last_row[0] = 0;
}

int main() {
    cl::Platform default_platform;
    cl::Device default_device;
    initialize(default_platform, default_device);

    auto kernel_code = load_kernel("pref_sum.cl");
    cl::Context context({default_device});
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);
    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << "Error building:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    std::ifstream fin("input.txt");
    size_t n;
    fin >> n;
    data_t* inp_arr = new data_t[n];
    for (size_t i = 0; i < n; ++i)
        fin >> inp_arr[i];

    size_t buff_size = sizeof(data_t) * n;
    size_t last_row_size = sizeof(data_t) * work_groups;
    cl::Buffer buffer_in(context, CL_MEM_READ_ONLY, buff_size);
    cl::Buffer buffer_last_row(context, CL_MEM_READ_WRITE, last_row_size);
    cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, buff_size);

    cl::CommandQueue queue(context, default_device);
    cl::EnqueueArgs eargs(queue, cl::NullRange, work_groups, 1);

    queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, buff_size, inp_arr);
    cl::Kernel first_kernel(program, "first_stage");
    cl::make_kernel<cl::Buffer&, size_t, cl::LocalSpaceArg, cl::Buffer&> first_stage(first_kernel);

    size_t m = std::max(work_groups, static_cast<size_t>(1 << static_cast<size_t>(std::ceil(std::log2(n)))));
    size_t local_memory_size = sizeof(data_t) * ((m / work_groups) * 2 - 1);

    std::cerr << "Needed memory: " << local_memory_size << std::endl;
    first_stage(eargs, buffer_in, n, cl::Local(local_memory_size), buffer_last_row).wait();

    data_t* last_row = new data_t[work_groups];
    queue.enqueueReadBuffer(buffer_last_row, CL_TRUE, 0, last_row_size, last_row);

    sum_last_row(last_row, work_groups);
    queue.enqueueWriteBuffer(buffer_last_row, CL_TRUE, 0, last_row_size, last_row);

    cl::Kernel second_kernel(program, "second_stage");
    cl::make_kernel<cl::Buffer&, cl::Buffer&, size_t, cl::LocalSpaceArg, cl::Buffer&> second_stage(second_kernel);
    second_stage(eargs, buffer_in, buffer_last_row, n, cl::Local(local_memory_size), buffer_out).wait();

    data_t* outp_arr = new data_t[n];
    queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, buff_size, outp_arr);

    std::ofstream fout("output.txt");
    for (size_t i = 0; i < n; ++i)
        fout << outp_arr[i] << " ";
    fout << std::endl;

    delete[] last_row;
    delete[] outp_arr;
    delete[] inp_arr;
}

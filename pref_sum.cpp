#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

#include <CL/cl.hpp>

const size_t work_items = 8;

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
    float* inp_arr = new float[n];
    for (size_t i = 0; i < n; ++i)
        fin >> inp_arr[i];

    size_t buff_size = sizeof(float) * n;
    cl::Buffer buffer_in(context, CL_MEM_READ_ONLY, buff_size);
    cl::Buffer buffer_out(context, CL_MEM_READ_WRITE, buff_size);

    cl::CommandQueue queue(context, default_device);
    queue.enqueueWriteBuffer(buffer_in, CL_TRUE, 0, buff_size, inp_arr);

    cl::Kernel kernel(program, "prefix_sum");
    cl::make_kernel<cl::Buffer&, size_t, cl::LocalSpaceArg, cl::Buffer&> prefix_sum(kernel);

    cl::EnqueueArgs eargs(queue, cl::NullRange, work_items, work_items);
    size_t local_memory_size = sizeof(float) *
            ((1 << static_cast<size_t>(std::ceil(std::log2(n))) + 1) - work_items);
    prefix_sum(eargs, buffer_in, n, cl::Local(local_memory_size), buffer_out).wait();

    float* outp_arr = new float[n];
    queue.enqueueReadBuffer(buffer_out, CL_TRUE, 0, buff_size, outp_arr);

    std::ofstream fout("output.txt");
    for (size_t i = 0; i < n; ++i)
        fout << outp_arr[i] << " ";
    fout << std::endl;

    delete[] outp_arr;
    delete[] inp_arr;
}

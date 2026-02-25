/*
 * MIT License
 *
 * Copyright (c) 2024 Fixstars Corp.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__x86_64__)
#include <cpuid.h>  // for __cpuid
#endif

#include <sys/mman.h>   // for mmap
#include <sys/types.h>  // for fork
#include <sys/wait.h>   // for waitpid
#include <unistd.h>     // for fork

#include "io_struct.hpp"

using io_struct::input_t;
using io_struct::output_t;

namespace io_struct
{
    void InitOutput(output_t &output);
    bool LoadInput(std::istream &is, input_t &input);
    bool StoreOutput(std::ostream &os, const output_t &output);
}  // namespace io_struct

void solve(input_t &input, output_t &output);

namespace
{
    /**
     * @brief Generate output_filepath from input_filepath
     *
     * @param input_filepath
     * @return std::string
     * @details For example:
     * @n ./foo/bar/in.txt -> ./foo/bar/out.txt
     * @n ./foo/bar/in -> ./foo/bar/out
     * @n ./foo/bar/in.in -> ./foo/bar/in.out
     * @n ./in/1.txt -> ./in/1.txt.out
     * @n abc -> abc.out
     */
    std::string GenOutputFilepath(const std::string &input_filepath)
    {
        // The first position of file name
        std::size_t last_slash_pos = input_filepath.find_last_of('/');
        if (last_slash_pos == std::string::npos)
        {
            last_slash_pos = 0;
        }
        else
        {
            last_slash_pos += 1;
        }
        std::string filename = input_filepath.substr(last_slash_pos);
        std::string from = "in";
        std::string to = "out";
        size_t p = filename.rfind(from);
        if (p != std::string::npos)
        {
            filename.replace(p, from.length(), to);
        }
        else
        {
            filename.append(".out");
        }
        const std::string output_filepath = input_filepath.substr(0, last_slash_pos) + filename;
        return output_filepath;
    }

#if defined(__x86_64__)
    unsigned int GetCacheLineSize()
    {
        unsigned int eax, ebx, ecx, edx;
        __cpuid(0x00000001, eax, ebx, ecx, edx);
        unsigned int cache_line_size = ((ebx & 0xFF00) >> 8) * 8;
        return cache_line_size;
    }

    inline void FlushCacheLine(void *addr) { asm volatile("clflush (%0)" ::"r"(addr)); }

    /**
     * @brief Clear cache only for x86_64 arch
     *
     * @param ptr
     * @param datasize
     */
    void ClearCache(char *ptr, size_t datasize)
    {
        const unsigned int cache_line_size = GetCacheLineSize();

        // = ceil(datasize/cache_line_size)
        const int num_of_flush = (datasize + (cache_line_size - 1)) / cache_line_size;

        for (int i = 0; i < num_of_flush; ++i)
        {
            FlushCacheLine(ptr + i * cache_line_size);
        }

        // for cases that struct isn't aligned to a 64-bit boundary
        FlushCacheLine(ptr + datasize - 1);
    }
#else
    void ClearCache([[maybe_unused]] char *ptr, [[maybe_unused]] size_t datasize)
    {
        std::cerr << "ClearCache isn't implemented except for x86_64" << std::endl;
    }
#endif

}  // namespace

int main(int argc, char *argv[])
{
    // Check arguments
    if (argc < 2)
    {
        const std::string project_name = argv[0];
        std::cerr << "Need input data path. (Example: " + project_name + " ../data/in0.txt)" << std::endl;
        return -1;
    }
    if (argc > 2)
    {
        const std::string project_name = argv[0];
        std::cerr << "Too many arguments. (Example: " + project_name + " ../data/in0.txt)" << std::endl;
        return -1;
    }

    // Shared memory of input/output data for the child process
    input_t *input_ptr =
        (input_t *)(mmap(0, sizeof(input_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0));
    output_t *output_ptr =
        (output_t *)(mmap(0, sizeof(output_t), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0));
    input_t &input = *input_ptr;
    output_t &output = *output_ptr;

    // Prepare input
    const std::string input_filepath = argv[1];
    std::ifstream ifs(input_filepath);
    if (io_struct::LoadInput(ifs, input) == false)
    {
        return -1;
    }
    ifs.close();

    // Initialize output
    io_struct::InitOutput(output);

    // Clear cache before fork
    ClearCache(reinterpret_cast<char *>(input_ptr), sizeof(input_t));
    ClearCache(reinterpret_cast<char *>(output_ptr), sizeof(output_t));

    pid_t pid = fork();
    if (pid == -1)
    {  // Fork failed
        std::cerr << "Fork failed" << std::endl;
        munmap(&output, sizeof(output_t));
        return -1;
    }

    if (pid == 0)
    {  // Child process
        solve(input, output);
        exit(0);
    }
    else
    {  // Parent process
        using namespace std::chrono;

        system_clock::time_point start, end;

        start = system_clock::now();
        int status;

        // Wait child process
        waitpid(pid, &status, 0);
        end = system_clock::now();

        // Check status of child process
        if (WIFSIGNALED(status))
        {
            std::cerr << "Error: solve() terminated with signal " << WTERMSIG(status) << std::endl;
            return WTERMSIG(status);
        }
        else if (!WIFEXITED(status))
        {
            std::cerr << "Error: solve() terminated abnormally without signal" << std::endl;
            return 1;
        }
        else
        {
            std::cerr << "Info: solve() terminated successfully" << std::endl;
        }

        // Get results
        const size_t elapsed = duration_cast<nanoseconds>(end - start).count();
        const std::string output_filepath = GenOutputFilepath(input_filepath);
        std::ofstream ofs(output_filepath);
        if (!ofs)
        {
            std::cerr << "Error: Failed to writing results into '" << output_filepath << "'" << std::endl;
            return 1;
        }
        ofs << "elapsed_nsec = " << elapsed << std::endl;
        io_struct::StoreOutput(ofs, output);
        ofs.close();
        std::cerr << "Info: Succeed in writing results into '" << output_filepath << "'" << std::endl;

        // Unmap input_ptr/output_ptr
        munmap(input_ptr, sizeof(input_t));
        munmap(output_ptr, sizeof(output_t));
    }
}

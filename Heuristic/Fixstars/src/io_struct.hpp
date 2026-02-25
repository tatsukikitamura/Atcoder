#ifndef IO_STRUCT_HPP_
#define IO_STRUCT_HPP_

#include <iostream>
#include <array>
#include <cstdint>

namespace io_struct
{
    // 問題に応じて調整する
    constexpr std::size_t PARAMETER_COUNT = 128;

    // N の最大値（問題では60だが余裕を持たせる）
    constexpr std::size_t MAX_N = 200;

    struct input_t
    {
        int N; // 実際の冒険者数 (<= MAX_N)

        // 隣接行列 A: N x N を MAX_N x MAX_N として確保
        std::array<std::array<int, MAX_N>, MAX_N> A;

        // 能力値 v: N x PARAMETER_COUNT
        std::array<std::array<int, PARAMETER_COUNT>, MAX_N> v;

        // BOSS 要求値 r: 1 x PARAMETER_COUNT
        std::array<int, PARAMETER_COUNT> r;
    };

    struct output_t
    {
        int K_size;  // パーティ人数
        long long T; // ParameterSum
        // 先頭 K_size 個だけ有効
        std::array<int, MAX_N> members;
    };

    inline void InitOutput(output_t &output)
    {
        output.K_size = 0;
        output.T = 0;
        // members は特にクリア不要（必要なら 0 埋めしてもよい）
        output.members.fill(0);
    }

    inline bool LoadInput(std::istream &is, input_t &input)
    {
        if (!is)
        {
            std::cerr << "Error: ifs isn't opened" << std::endl;
            return false;
        }

        // N
        is >> input.N;
        if (input.N <= 0 || input.N > static_cast<int>(MAX_N))
        {
            std::cerr << "Error: Invalid N value: " << input.N
                      << " (MAX_N = " << MAX_N << ")" << std::endl;
            return false;
        }

        const int N = input.N;

        // A (N x N)
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                is >> input.A[i][j];
            }
        }

        // v (N x PARAMETER_COUNT)
        for (int i = 0; i < N; ++i)
        {
            for (std::size_t k = 0; k < PARAMETER_COUNT; ++k)
            {
                is >> input.v[i][k];
            }
        }

        // r (1 x PARAMETER_COUNT)
        for (std::size_t k = 0; k < PARAMETER_COUNT; ++k)
        {
            is >> input.r[k];
        }

        return true;
    }

    inline bool StoreOutput(std::ostream &os, const output_t &output)
    {
        if (!os)
        {
            std::cerr << "Error: os isn't opened" << std::endl;
            return false;
        }

        os << "PartySize = " << output.K_size << std::endl;
        os << "ParameterSum = " << output.T << std::endl;

        os << "PartyMembers = {\n";
        for (int i = 0; i < output.K_size; ++i)
        {
            if (i > 0)
            {
                os << " ";
            }
            os << output.members[i]; // 1-indexed, 昇順想定
        }
        os << "\n}" << std::endl;

        return true;
    }

} // namespace io_struct

#endif // IO_STRUCT_HPP_

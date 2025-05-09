#include "delta_filter.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

namespace dz
{
    std::size_t deltaFilter(const uint8_t* d_flag, std::size_t N, thrust::device_vector<int>& outIdx, cudaStream_t stream)
    {
        using thrust::make_counting_iterator;
        using thrust::device_pointer_cast;
        using thrust::cuda::par;

        // 0..N-1 lazy counting iterator
        auto first = make_counting_iterator<int>(0); //auto로 타입 자동 입력
        auto last  = first + static_cast<int>(N);

        outIdx.resize(N);   // 최대 크기 예약

        // flag!=0 인 인덱스만 복사
        auto endIt = thrust::copy_if(par.on(stream), first, last, device_pointer_cast(d_flag), outIdx.begin(), thrust::identity<int>());

        std::size_t cnt = endIt - outIdx.begin();
        outIdx.resize(cnt);   // 실제 크기로 축소
        return cnt;
    }

}
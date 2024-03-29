#version 450 core

// https://poniesandlight.co.uk/reflect/bitonic_merge_sort/

#define eLocalBitonicMergeSortExample 0
#define eLocalDisperse 1
#define eBigFlip 2
#define eBigDisperse 3

layout(local_size_x=1024, local_size_y=1, local_size_z=1) in;

layout(std430, binding=0) buffer sort_data
{
    // This is our unsorted input buffer - tightly packed,
    // an array of N_GLOBAL_ELEMENTS elements.
    uint value[];
};

// uniform uint algorithm;
uniform uint u_h;
uniform uint u_algorithm;

// Workgroup local memory. We use this to minimise round-trips to global memory.
// It allows us to evaluate a sorting network of up to 1024 with one shader invocation.
shared uint local_value[gl_WorkGroupSize.x * 2];

// naive comparison
bool is_greater(in const uint left, in const uint right){
    return left > right;
}

// Pick comparison funtion
#define COMPARE is_greater

#define SWAP(T, x, y) { T tmp = x; x = y; y = tmp; }

void global_compare_and_swap(ivec2 idx){
    if (COMPARE(value[idx.x], value[idx.y])) {
        SWAP(uint, value[idx.x], value[idx.y]);
    }
}

void local_compare_and_swap(ivec2 idx){
    if (COMPARE(local_value[idx.x], local_value[idx.y])) {
        SWAP(uint, local_value[idx.x], local_value[idx.y]);
    }
}

// Performs full-height flip (h height) over globally available indices.
void big_flip(in uint h) {

    uint t_prime = gl_GlobalInvocationID.x;
    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2

    uint q       = ((2 * t_prime) / h) * h;
    uint x       = q     + (t_prime % half_h);
    uint y       = q + h - (t_prime % half_h) - 1;

    global_compare_and_swap(ivec2(x,y));
}

// Performs full-height disperse (h height) over globally available indices.
void big_disperse(in uint h) {

    uint t_prime = gl_GlobalInvocationID.x;

    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2

    uint q       = ((2 * t_prime) / h) * h;
    uint x       = q + (t_prime % (half_h));
    uint y       = q + (t_prime % (half_h)) + half_h;

    global_compare_and_swap(ivec2(x,y));

}

// Performs full-height flip (h height) over locally available indices.
void local_flip(in uint h){
    uint t = gl_LocalInvocationID.x;
    barrier();

    uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2
    ivec2 indices =
        ivec2( h * ( ( 2 * t ) / h ) ) +
        ivec2( t % half_h, h - 1 - ( t % half_h ) );

    local_compare_and_swap(indices);
}

// Performs progressively diminishing disperse operations (starting with height h)
// on locally available indices: e.g. h==8 -> 8 : 4 : 2.
// One disperse operation for every time we can divide h by 2.
void local_disperse(in uint h){
    uint t = gl_LocalInvocationID.x;
    for ( ; h > 1 ; h /= 2 ) {

        barrier();

        uint half_h = h >> 1; // Note: h >> 1 is equivalent to h / 2
        ivec2 indices =
            ivec2( h * ( ( 2 * t ) / h ) ) +
            ivec2( t % half_h, half_h + ( t % half_h ) );

        local_compare_and_swap(indices);
    }
}

void local_bitonic_merge_sort_example(uint h){
    uint t = gl_LocalInvocationID.x;
    for ( uint hh = 2; hh <= h; hh <<= 1 ) {  // note:  h <<= 1 is same as h *= 2
        local_flip( hh);
        local_disperse( hh/2 );
    }
}

void main(){
    const uint t = gl_LocalInvocationID.x;

    const uint offset = gl_WorkGroupSize.x * 2 * gl_WorkGroupID.x; // we can use offset if we have more than one invocation.

    if (u_algorithm <= eLocalDisperse){
        // In case this shader executes a `local_` algorithm, we must
        // first populate the workgroup's local memory.

        local_value[t*2]   = value[offset+t*2];
        local_value[t*2+1] = value[offset+t*2+1];
    }

    switch (u_algorithm) {
        case eLocalBitonicMergeSortExample:
            local_bitonic_merge_sort_example(u_h);
        break;
        case eLocalDisperse:
            local_disperse(u_h);
        break;
        case eBigFlip:
            big_flip(u_h);
        break;
        case eBigDisperse:
            big_disperse(u_h);
        break;
    }

    // Write local memory back to buffer in case we pulled in the first place.
    if (u_algorithm <= eLocalDisperse){
        barrier();
        // push to global memory
        value[offset+t*2]   = local_value[t*2];
        value[offset+t*2+1] = local_value[t*2+1];
    }

}

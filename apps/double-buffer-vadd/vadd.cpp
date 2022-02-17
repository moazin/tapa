#include <cstdint>
#include <tapa.h>

using data_t = int;
using addr_t = unsigned int;
#define BUFFER_SIZE 32
#define ITER 16


void WriteBack(tapa::istream<data_t>& c_out, tapa::mmap<data_t> c) {
  for (addr_t i = 0; i < BUFFER_SIZE * ITER; ++i) {
    c[i] = c_out.read();
  }
}

void ComputePart2(
    tapa::istream<data_t>& a_in, 
    tapa::istream<data_t>& b_in, 
    tapa::ostream<data_t>& c_out
) {
  data_t temp;
  for (addr_t i = 0; i < BUFFER_SIZE * ITER; ++i) {
    #pragma HLS pipeline II=1
    c_out.write(a_in.read() + b_in.read());
  }
}

void _compute_part1(
  data_t buffer[BUFFER_SIZE],
  tapa::ostream<data_t> &dout
) {
  for (int i = 0; i < BUFFER_SIZE; i++) {
    dout.write(buffer[i]);
  }
}

void _load(
    tapa::mmap<data_t> mmap,
    data_t buffer[BUFFER_SIZE],
    addr_t offset
) {
  for (int i = 0; i < BUFFER_SIZE; i++) {
    #pragma HLS pipeline II=1
    buffer[i] = mmap[offset + i];
  }
}

void DoubleBuffer(
    tapa::mmap<data_t> mmap,
    tapa::ostream<data_t> &dout
) {
  data_t buffers[2][BUFFER_SIZE];
  #pragma HLS array_partition variable=buffers dim=1 complete

  for (int i = 0; i < ITER; i++) {
    if (i%2) {
      _load(mmap, buffers[i%2], i * BUFFER_SIZE);
      _compute_part1(buffers[i%2+1], dout);
    }
    else {
      _load(mmap, buffers[i%2+1], i * BUFFER_SIZE);
      _compute_part1(buffers[i%2], dout);
    }
  }
}

void VecAdd(
    tapa::mmap<data_t> a, 
    tapa::mmap<data_t> b, 
    tapa::mmap<data_t> c
) {
  tapa::stream<data_t, 2> a_read;
  tapa::stream<data_t, 2> b_read;
  tapa::stream<data_t, 2> c_write;

  tapa::task()
    .invoke(DoubleBuffer, a, a_read)
    .invoke(DoubleBuffer, b, b_read)
    .invoke(ComputePart2, a_read, b_read, c_write)
    .invoke(WriteBack, c_write, c)
  ;
}

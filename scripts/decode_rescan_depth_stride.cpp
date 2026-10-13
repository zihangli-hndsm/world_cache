// Decode every Nth Rescan depth frame using the official ScannerApp codec.
// Codec headers are archived at outputs/reports/20261008_rescan_native_inventory/.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "uplinksimple_image-codecs.h"
#include "uplinksimple_shift2depth.h"

static void write_pgm(const std::filesystem::path& path,
                      const std::vector<uint16_t>& depth,
                      size_t width, size_t height) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot create " + path.string());
  out << "P5\n# uint16 depth in millimeters\n" << width << " " << height << "\n65535\n";
  for (uint16_t value : depth) {
    const char bytes[2] = {static_cast<char>((value >> 8) & 0xff),
                           static_cast<char>(value & 0xff)};
    out.write(bytes, 2);
  }
  if (!out) throw std::runtime_error("short write for " + path.string());
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: decode_rescan_depth_stride INPUT.depth OUTPUT_DIR STRIDE\n";
    return 2;
  }
  const std::filesystem::path input_path(argv[1]);
  const std::filesystem::path output_dir(argv[2]);
  const int stride = std::stoi(argv[3]);
  if (stride < 1 || !std::filesystem::is_directory(output_dir)) {
    std::cerr << "stride must be positive and output directory must exist\n";
    return 2;
  }
  std::ifstream in(input_path, std::ios::binary);
  if (!in) { std::cerr << "cannot open input\n"; return 2; }
  in.seekg(0, std::ios::end);
  const uint64_t total_bytes = static_cast<uint64_t>(in.tellg());
  in.seekg(0, std::ios::beg);

  constexpr size_t width = 640, height = 480;
  size_t frame_index = 0, saved_count = 0;
  std::vector<uint16_t> depth(width * height);
  while (true) {
    uint32_t payload_size = 0;
    in.read(reinterpret_cast<char*>(&payload_size), sizeof(payload_size));
    if (in.eof() && in.gcount() == 0) break;
    if (!in || payload_size == 0 || payload_size > 1'000'000) {
      std::cerr << "invalid depth frame header at " << frame_index << "\n";
      return 3;
    }
    std::vector<uint8_t> payload(payload_size);
    in.read(reinterpret_cast<char*>(payload.data()), payload_size);
    if (!in) { std::cerr << "truncated payload at " << frame_index << "\n"; return 3; }

    if (frame_index % static_cast<size_t>(stride) == 0) {
      uplinksimple::decode(payload.data(), payload_size, depth.size(), depth.data());
      uplinksimple::shift2depth(depth.data(), depth.size());
      const uint16_t max_depth = uplinksimple::shift2depth(0xffff);
      for (uint16_t& value : depth) if (value >= max_depth) value = 0;
      const auto path = output_dir / ("frame_" + std::to_string(saved_count) + ".pgm");
      write_pgm(path, depth, width, height);
      ++saved_count;
    }
    ++frame_index;
    const uint64_t remaining = total_bytes - static_cast<uint64_t>(in.tellg());
    if (remaining == frame_index * 16) break;  // two double timestamp arrays
  }
  std::cout << "decoded_frames=" << frame_index << " saved_frames=" << saved_count
            << " stride=" << stride << " timestamp_bytes_remaining="
            << (total_bytes - static_cast<uint64_t>(in.tellg())) << "\n";
  return 0;
}

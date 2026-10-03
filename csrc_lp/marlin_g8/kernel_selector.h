if (false) {}
#ifndef LP_NO_S8
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kS8.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
#endif
#ifdef LP_WITH_FP16
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 4, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 1, 8, 4, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 2, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 3, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 4, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 128, 4, 8, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && m_block_size_8 == true && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 1, 8, 8, true, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 2, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 3, 16, 4, false, 2, 8, false>;
else if (a_type == vllm::kFloat16 && b_type == vllm::kU4 && c_type == vllm::kFloat16 && s_type == vllm::kFloat16 && threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 && thread_k_blocks == 4 && m_block_size_8 == false && stages == 2 && group_blocks == 8 && is_zp_float == false)
  kernel = Marlin<vllm::kFloat16.id(), vllm::kU4.id(), vllm::kFloat16.id(), vllm::kFloat16.id(), 256, 4, 16, 4, false, 2, 8, false>;
#endif

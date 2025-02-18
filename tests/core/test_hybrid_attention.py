import unittest
import torch
from danon.core.attention import (
    HybridConfig,
    HybridAttentionModel,
    create_hybrid_model,
    MSRAModel,
    DALAModel,
    UCSAModel
)

class TestHybridAttention(unittest.TestCase):
    def setUp(self):
        self.config = HybridConfig(
            hidden_size=256,
            num_levels=2,
            chunk_size=64,
            num_layers=2,
            auto_switch_threshold=100,  # 设置较小的阈值便于测试
            enable_hybrid_mode=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_model_creation(self):
        """测试模型创建"""
        model = create_hybrid_model(
            hidden_size=256,
            num_levels=2,
            num_layers=2,
            auto_switch_threshold=100
        )
        self.assertIsInstance(model, HybridAttentionModel)
        
    def test_msra_mode(self):
        """测试MSRA模式（短序列）"""
        model = HybridAttentionModel(self.config).to(self.device)
        batch_size, seq_len = 2, 50  # 小于阈值的序列长度
        input_ids = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 测试基本功能
        output, mode_info = model(input_ids, return_mode_info=True)
        self.assertEqual(output.shape, input_ids.shape)
        self.assertEqual(mode_info['mode'], 'MSRA')
        
    def test_dala_mode(self):
        """测试DALA模式（长序列）"""
        model = HybridAttentionModel(self.config).to(self.device)
        batch_size, seq_len = 2, 150  # 大于阈值的序列长度
        input_ids = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 测试基本功能
        output, mode_info = model(input_ids, return_mode_info=True)
        self.assertEqual(output.shape, input_ids.shape)
        self.assertEqual(mode_info['mode'], 'DALA')
        
    def test_ucsa_mode(self):
        """测试UCSA模式（超长序列）"""
        model = HybridAttentionModel(self.config).to(self.device)
        batch_size, seq_len = 2, 1000  # 超长序列
        input_ids = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 测试基本功能
        output, mode_info = model(input_ids, return_mode_info=True)
        self.assertEqual(output.shape, input_ids.shape)
        self.assertEqual(mode_info['mode'], 'UCSA')
        
    def test_mode_switching(self):
        """测试模式切换"""
        model = HybridAttentionModel(self.config).to(self.device)
        
        # 短序列 -> MSRA
        short_input = torch.randn(2, 50, self.config.hidden_size).to(self.device)
        _, short_info = model(short_input, return_mode_info=True)
        self.assertEqual(short_info['mode'], 'MSRA')
        
        # 长序列 -> DALA
        long_input = torch.randn(2, 150, self.config.hidden_size).to(self.device)
        _, long_info = model(long_input, return_mode_info=True)
        self.assertEqual(long_info['mode'], 'DALA')
        
    def test_disable_hybrid_mode(self):
        """测试禁用混合模式"""
        config = HybridConfig(
            hidden_size=256,
            enable_hybrid_mode=False
        )
        model = HybridAttentionModel(config).to(self.device)
        
        # 即使是长序列也应该使用MSRA
        input_ids = torch.randn(2, 150, config.hidden_size).to(self.device)
        _, mode_info = model(input_ids, return_mode_info=True)
        self.assertEqual(mode_info['mode'], 'MSRA')
        
    def test_gradient_flow(self):
        """测试梯度流动"""
        model = HybridAttentionModel(self.config).to(self.device)
        
        # 测试MSRA模式的梯度
        short_input = torch.randn(2, 50, self.config.hidden_size, requires_grad=True).to(self.device)
        short_output, _ = model(short_input, return_mode_info=True)
        short_loss = short_output.sum()
        short_loss.backward()
        self.assertIsNotNone(short_input.grad)
        
        # 测试DALA模式的梯度
        model.zero_grad()
        long_input = torch.randn(2, 150, self.config.hidden_size, requires_grad=True).to(self.device)
        long_output, _ = model(long_input, return_mode_info=True)
        long_loss = long_output.sum()
        long_loss.backward()
        self.assertIsNotNone(long_input.grad)

if __name__ == '__main__':
    unittest.main() 
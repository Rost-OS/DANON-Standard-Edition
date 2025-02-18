import unittest
import torch
import torch.nn as nn
from danon.core.attention import (
    MSRAConfig,
    EnhancedSelfCalibratingAttention,
    CrossLayerFeatureEnhancer,
    MSRAEnhancedLayer
)

class TestAttentionMechanism(unittest.TestCase):
    def setUp(self):
        self.config = MSRAConfig(
            hidden_size=256,
            num_levels=3,
            chunk_size=64,
            compression_factor=4,
            dropout=0.1,
            num_layers=4
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def test_enhanced_self_calibrating_attention(self):
        model = EnhancedSelfCalibratingAttention(self.config).to(self.device)
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 基本功能测试
        output, meta = model(x)
        self.assertEqual(output.shape, x.shape)
        
        # 稳定性测试
        for _ in range(5):
            out1, _ = model(x)
            out2, _ = model(x)
            diff = (out1 - out2).abs().mean()
            self.assertLess(diff, 1e-6)
            
    def test_cross_layer_feature_enhancer(self):
        model = CrossLayerFeatureEnhancer(self.config).to(self.device)
        batch_size, seq_len = 4, 32
        current = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        prev = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        next_features = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        
        # 基本功能测试
        output = model(current, prev, next_features)
        self.assertEqual(output.shape, current.shape)
        
        # 边界条件测试
        output_no_prev = model(current, None, next_features)
        self.assertEqual(output_no_prev.shape, current.shape)
        
    def test_msra_enhanced_layer(self):
        model = MSRAEnhancedLayer(self.config).to(self.device)
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size).to(self.device)
        attention_mask = torch.ones(batch_size, seq_len).to(self.device)
        
        # 基本功能测试
        output, meta = model(hidden_states, attention_mask)
        self.assertEqual(output.shape, hidden_states.shape)
        
        # 梯度检查
        hidden_states.requires_grad = True
        output, _ = model(hidden_states, attention_mask)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        
if __name__ == '__main__':
    unittest.main() 
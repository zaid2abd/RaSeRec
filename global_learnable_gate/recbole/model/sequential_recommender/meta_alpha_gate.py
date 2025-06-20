import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAlphaFusionGate(nn.Module):
    """
    Meta-α Fusion Gate مع معاملات قابلة للتعلم عامة
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super(MetaAlphaFusionGate, self).__init__()
        self.hidden_size = hidden_size
        
        # معاملات قابلة للتعلم
        self.alpha_param = nn.Parameter(torch.tensor(0.42))  # sigmoid(0.42) ≈ 0.603
        self.beta_param = nn.Parameter(torch.tensor(-0.42))  # sigmoid(-0.42) ≈ 0.397

        print(f"🔧 Initialized Learnable Meta-α Fusion Gate")
        print(f"🔧 Initial params: α={torch.sigmoid(self.alpha_param).item():.4f}, β={torch.sigmoid(self.beta_param).item():.4f}")
        
    def forward(self, user_repr, channel1_repr, channel2_repr, retrieved_memories):
        batch_size = user_repr.size(0)
        
        # إجبار requires_grad=True
        self.alpha_param.requires_grad_(True)
        self.beta_param.requires_grad_(True)
        
        # استخدام parameters قابلة للتعلم
        adaptive_alpha = torch.sigmoid(self.alpha_param).expand(batch_size, 1)
        adaptive_beta = torch.sigmoid(self.beta_param).expand(batch_size, 1)
        
        # الدمج باستخدام المعاملات المتعلمة (النسخة الأصلية)
        fused_representation = (
            adaptive_alpha * user_repr + 
            (1 - adaptive_alpha) * (
                adaptive_beta * channel1_repr + 
                (1 - adaptive_beta) * channel2_repr
            )
        )
        
        # التأكد من أن النتيجة تحتفظ بـ gradients
        fused_representation = fused_representation.requires_grad_(True)
        
        print(f"🔍 Meta-α: α={torch.sigmoid(self.alpha_param).item():.4f}, β={torch.sigmoid(self.beta_param).item():.4f}")
        
        return fused_representation, {
            'adaptive_alpha': adaptive_alpha,
            'adaptive_beta': adaptive_beta,
            'memory_relevance': torch.ones_like(adaptive_alpha)
        }

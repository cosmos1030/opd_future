import torch, torch.nn.functional as F
torch.manual_seed(0)
s = F.log_softmax(torch.randn(4, 7, 11).double(), dim=-1)
t = F.log_softmax(torch.randn(4, 7, 11).double(), dim=-1)

ce_new = -(t.exp() * s).sum(-1)
rev_new = F.kl_div(t, s, reduction="none", log_target=True).sum(-1)
rev_ref = (s.exp() * (s - t)).sum(-1)
print("revkl matches closed form :", torch.allclose(rev_new, rev_ref))
print("revkl >= 0 everywhere     :", bool((rev_new >= -1e-12).all()))
print("ce    >= 0 everywhere     :", bool((ce_new >= 0).all()))
print("revkl mean %.4f   ce mean %.4f" % (rev_new.mean(), ce_new.mean()))

# the sdpo path's alpha=1.0 branch, for comparison
sdpo_rev = F.kl_div(t, s, reduction="none", log_target=True).sum(-1)
print("same as sdpo alpha=1.0    :", torch.allclose(rev_new, sdpo_rev))

# forward KL branch, and the CE = KL + H(pi_T) identity it rests on
fwd_new = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
fwd_ref = (t.exp() * (t - s)).sum(-1)
H_T     = -(t.exp() * t).sum(-1)
print("fwdkl matches closed form :", torch.allclose(fwd_new, fwd_ref))
print("fwdkl >= 0 everywhere     :", bool((fwd_new >= -1e-12).all()))
print("ce == fwdkl + H(pi_T)     :", torch.allclose(ce_new, fwd_new + H_T))
print("fwdkl mean %.4f   H mean %.4f   ce mean %.4f" % (fwd_new.mean(), H_T.mean(), ce_new.mean()))

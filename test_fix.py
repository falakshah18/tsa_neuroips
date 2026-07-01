import logging, io, torch
from spikingjelly.activation_based import functional
from baselines.temporal_coding.ttfs_snn import TTFSNetwork_Vision, TTFSNetwork
from baselines.eprop.eprop_snn import EpropSNN, EpropSNN_Vision

# Capture warnings so we can check none were logged
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.WARNING)

# TTFS vision
m = TTFSNetwork_Vision(in_channels=2, num_classes=10, img_size=34, T_encode=5)
out, metrics = m(torch.rand(5, 2, 2, 34, 34))
functional.reset_net(m)
print("TTFSNetwork_Vision OK:", out.shape)

# TTFS SHD-style
m2 = TTFSNetwork(input_size=64, hidden_size=32, num_classes=5, T_encode=5)
out2, metrics2 = m2(torch.rand(4, 64))
functional.reset_net(m2)
print("TTFSNetwork OK:", out2.shape)

# E-prop
m3 = EpropSNN(input_size=64, hidden_size=32, num_classes=5)
out3, metrics3 = m3(torch.rand(3, 4, 64))
functional.reset_net(m3)
print("EpropSNN OK:", out3.shape)

warnings_captured = log_stream.getvalue()
if "not spikingjelly" in warnings_captured:
    print("❌ FAILURE — warnings still present:")
    print(warnings_captured[:1500])
else:
    print("✅ SUCCESS — no MemoryModule warnings")
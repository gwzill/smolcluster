import torch
from smolcluster.models.moe import Router, Mixtral

# Test 1: Create Mixtral WITHOUT passing a router (default behavior)
print('Test 1: Default behavior (no router passed)')
model1 = Mixtral(vocab_size=1000, embeddings_dims=128, no_of_decoder_layers=2, num_experts=4, top_k=2)
print('  Created Mixtral with internally created routers')
print(f'  Each layer has its own router: {model1.decoder_layers[0].moe_block.router is not model1.decoder_layers[1].moe_block.router}')

# Test 2: Create a shared Router and pass it to Mixtral
print('\nTest 2: Pass custom router (shared across layers)')
device = torch.device('cpu')
shared_router = Router(embeddings_dims=128, num_experts=4, top_k=2, device=device, noisy_topk=True)
model2 = Mixtral(vocab_size=1000, embeddings_dims=128, no_of_decoder_layers=2, num_experts=4, top_k=2, router=shared_router)
print('  Created Mixtral with shared router')
print(f'  All layers share the same router: {model2.decoder_layers[0].moe_block.router is model2.decoder_layers[1].moe_block.router}')
print(f'  Router is the same as the one we passed: {model2.decoder_layers[0].moe_block.router is shared_router}')

# Test 3: Run a forward pass to verify it works
print('\nTest 3: Forward pass with custom router')
input_ids = torch.randint(0, 1000, (2, 10))
output = model2(input_ids)
print(f'  Input shape: {input_ids.shape}')
print(f'  Output shape: {output.shape}')
print('  ✅ Forward pass successful!')

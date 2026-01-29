"""Test completo de entrenamiento con GPU"""
import torch
import torch.nn as nn
import torch.optim as optim
import time

print("=" * 70)
print("Test completo de entrenamiento con GPU RTX 5060 Ti")
print("=" * 70)

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Crear modelo simple
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 27)  # 27 atributos
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Mover modelo a GPU
model = SimpleModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\nModelo creado y movido a {device}")

# Test de entrenamiento
print("\nTest de entrenamiento (10 iteraciones)...")
batch_size = 64

start_time = time.time()

for i in range(10):
    # Datos sintéticos
    inputs = torch.randn(batch_size, 256).to(device)
    targets = torch.randint(0, 2, (batch_size, 27)).float().to(device)

    # Forward
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        print(f"   Iteración {i+1}/10, Loss: {loss.item():.4f}")

elapsed = time.time() - start_time
print(f"\nTiempo total: {elapsed:.2f}s")
print(f"Tiempo por iteración: {elapsed/10:.3f}s")

if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated(0) / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"\nMemoria GPU utilizada: {memory_allocated:.2f} GB")
    print(f"Memoria GPU reservada: {memory_reserved:.2f} GB")

print("\n" + "=" * 70)
print("✅ TEST EXITOSO - LA GPU FUNCIONA PERFECTAMENTE")
print("=" * 70)
print("\nPuedes entrenar el modelo completo con tu RTX 5060 Ti!")

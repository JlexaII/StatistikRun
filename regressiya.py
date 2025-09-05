# regressiya.py
import numpy as np
import pandas as pd

# ==============================
# 1. Ma'lumotlarni tayyorlash
# ==============================
# Jadvalni misol uchun yaratamiz (siz CSV yoki Excel'dan ham yuklashingiz mumkin)
# Y - natija, X1, X2, X3 - mustaqil o'zgaruvchilar
data = {
    "Y":  [10, 12, 13, 15, 16, 18, 20, 22, 25, 30],
    "X1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "X2": [2, 1, 4, 3, 5, 7, 6, 8, 9, 10],
    "X3": [5, 3, 6, 7, 8, 10, 11, 13, 15, 18]
}
df = pd.DataFrame(data)
print("Ma'lumotlar jadvali:")
print(df)

# ==============================
# 2. Matritsalarni tayyorlash
# ==============================
Y = df["Y"].values.reshape(-1, 1)       # (n x 1)
X = df[["X1", "X2", "X3"]].values       # (n x k)

# X ga 1-lar ustuni qo'shamiz (b0 uchun)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# ==============================
# 3. Koeffitsientlarni hisoblash
# ==============================
# Formulaga ko'ra: b = (X^T * X)^-1 * X^T * Y
XTX = X.T @ X
XTY = X.T @ Y
B = np.linalg.inv(XTX) @ XTY

print("\nRegressiya koeffitsientlari (b0, b1, b2, b3):")
print(B.flatten())

# ==============================
# 4. Bashorat qilingan qiymatlar va qoldiq
# ==============================
Y_hat = X @ B
residuals = Y - Y_hat

# ==============================
# 5. Statistik ko'rsatkichlar
# ==============================
# SSR (Regression Sum of Squares)
SSR = np.sum((Y_hat - Y.mean())**2)
# SSE (Error Sum of Squares)
SSE = np.sum((Y - Y_hat)**2)
# SST (Total Sum of Squares)
SST = np.sum((Y - Y.mean())**2)

# Determinatsiya koeffitsienti
R2 = SSR / SST
R = np.sqrt(R2)

# Standart xatolik (sigma)
n, k = X.shape
sigma2 = SSE / (n - k)
var_b = sigma2 * np.linalg.inv(XTX)
std_errors = np.sqrt(np.diag(var_b))

print("\nStatistik ko'rsatkichlar:")
print(f"SST = {SST:.4f}, SSR = {SSR:.4f}, SSE = {SSE:.4f}")
print(f"R^2 = {R2:.4f}, R = {R:.4f}")
print(f"Standart xatoliklar (SE): {std_errors}")

# ==============================
# 6. Natijaviy tenglama
# ==============================
eq = f"Y = {B[0][0]:.4f}"
for i in range(1, len(B)):
    eq += f" + ({B[i][0]:.4f})*X{i}"

print("\nRegressiya tenglamasi:")
print(eq)

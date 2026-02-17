# import numpy as np
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# BUDGET = 943_167_086.78

# REDIST_PER_HA = (0, 0)

# yfs_values = np.arange(0, 150, 1)
# roots = []

# for yfs in yfs_values:
#     solution = find_flat_rate(data, BUDGET, yfs, REDIST_PER_HA_01)
#     roots.append(solution.root)

# # --- Plot ---
# plt.figure(figsize=(6, 4))
# plt.plot(yfs_values, roots, marker="o", linestyle="-", color="C0")
# plt.title("Relationship between YFS and root")
# plt.xlabel("YFS")
# plt.ylabel("Root")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # arrays (drop NaNs if any)
# x = np.asarray(yfs_values, dtype=float)
# y = np.asarray(roots, dtype=float)
# mask = np.isfinite(x) & np.isfinite(y)
# x, y = x[mask], y[mask]

# # OLS: root = beta0 + beta1 * YFS
# X = sm.add_constant(x)  # adds intercept
# model = sm.OLS(y, X).fit()
# print(model.summary())

# # predictions on a dense grid for a smooth line
# x_pred = np.linspace(x.min(), x.max(), 200)
# X_pred = sm.add_constant(x_pred)
# y_pred = model.predict(X_pred)

# # plot data + fitted line
# plt.figure(figsize=(6, 4))
# plt.scatter(x, y)  # points
# plt.plot(x_pred, y_pred)  # fitted line
# plt.title("Root vs YFS with OLS fit")
# plt.xlabel("YFS")
# plt.ylabel("Root")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# yfs_pred = np.linspace(yfs_values.min(), yfs_values.max(), 100)
# budg_pred = np.linspace(budget_values.min(), budget_values.max(), 100)
# Bp, Yp = np.meshgrid(budg_pred, yfs_pred, indexing="ij")

# df_pred = pd.DataFrame({"YFS": Yp.ravel(), "Budget": Bp.ravel()})
# Zhat = model.predict(df_pred).to_numpy().reshape(Bp.shape)

# plt.figure(figsize=(7,5))
# cont = plt.contourf(Yp, Bp, Zhat, levels=20, cmap="Blues")
# lines = plt.contour(Yp, Bp, Zhat, levels=20, colors="black", linewidths=0.7)
# plt.clabel(lines, inline=True, fmt="%.1f", fontsize=8)
# plt.colorbar(cont, label="Fitted root")
# plt.xlabel("YFS")
# plt.ylabel("Budget")
# plt.title("Fitted root ~ YFS + Budget (OLS)")
# plt.tight_layout(); plt.show()

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# from statsmodels.formula.api import ols

# BUDGET = 943_167_086.78

# REDIST_PER_HA = (0, 0)

# yfs_values = np.arange(0, 150, 10)
# budget_values = np.arange(900_000_000, 1_000_000_000, 5_000_000)
# roots = []

# roots = np.zeros((len(budget_values), len(yfs_values)))

# for i, budget in enumerate(budget_values):
#     for j, yfs in enumerate(yfs_values):
#         solution = find_flat_rate(data, budget, yfs, REDIST_PER_HA)
#         roots[i, j] = solution.root

# # --- Plot ---
# B, Y = np.meshgrid(budget_values, yfs_values, indexing="ij")

# plt.figure(figsize=(7, 5))
# levels = np.linspace(roots.min(), roots.max(), 15)

# contour = plt.contourf(Y, B, roots, levels=levels, cmap="Blues", extend="both")
# lines = plt.contour(Y, B, roots, levels=levels, colors="black", linewidths=0.7)
# plt.clabel(lines, inline=True, fmt="%.1f", fontsize=8)

# plt.colorbar(contour, label="Root")
# plt.xlabel("YFS")
# plt.ylabel("Budget")
# plt.title("Root as a function of YFS and Budget")
# plt.tight_layout()
# plt.show()

# df_fit = pd.DataFrame({
#     "YFS":    Y.ravel(),        # from np.meshgrid
#     "Budget": B.ravel(),
#     "root":   roots.ravel(),
# }).dropna()

# # OLS: root = β0 + β1*YFS + β2*Budget
# model = ols("root ~ YFS + Budget", data=df_fit).fit()
# print(model.summary())

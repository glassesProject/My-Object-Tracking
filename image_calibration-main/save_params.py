import numpy as np

# これは calibrateCamera の出力から得たものとします
# ここでは仮の例。実際にはあなたの main.py で得られた mtx, dist を使ってください
mtx = [[2.40245832e+03, 0.00000000e+00, 6.74213181e+02],
       [0.00000000e+00, 2.41162074e+03, 5.94492481e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist =  [[-9.05537758e-01, -2.08369348e+01, -1.81662756e-03, -6.74583648e-03,
1.23594739e+02]]

# NumPy形式で保存（camera_params.npz というファイルができます）
np.savez("camera_params.npz", mtx=mtx, dist=dist)

print("✅ カメラパラメータを camera_params.npz に保存しました。")

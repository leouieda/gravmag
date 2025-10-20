import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import verde as vd
import harmonica as hm


def plota_modelo(x, dados, modelo, predito, mesh):    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True)
    axes[0].plot(x, dados, "o", label="observado")
    axes[0].plot(x, predito, label="predito", linewidth=2)
    axes[0].legend()
    scale = vd.maxabs(mesh.density)
    mesh.density.plot(
        ax=axes[1], vmin=-scale, vmax=scale, 
        cmap="RdBu_r", 
        cbar_kwargs={"aspect": 30, "pad": 0, "orientation": "vertical", "label": "kg/m³"},
    )
    for prism in modelo:
        axes[1].plot(
            [prism[0], prism[1], prism[1], prism[0], prism[0]],
            [prism[5], prism[5], prism[4], prism[4], prism[5]],
            color="yellow",
            linewidth=4,
        )
    return fig, axes


def inversao_linear(coordenadas, dados, tamanho_celula, damping=0, potencia=0, smoothness=0):
    # Constroi a malha
    mesh_coords = vd.grid_coordinates(
        region=(coordenadas[0].min(), coordenadas[0].max(), -15e3, 0),
        spacing=tamanho_celula,
        pixel_register=True,
    )
    mesh = vd.make_xarray_grid(
        coordinates=mesh_coords,
        data=np.zeros_like(mesh_coords[0]),
        data_names="density",
        dims=("upward", "horizontal"),
    )

    # Constroi a matriz de sensibilidade
    A = np.empty((dados.size, mesh.density.size))
    size_horizontal = (mesh.horizontal[1] - mesh.horizontal[0]) / 2
    size_vertical = (mesh.upward[1] - mesh.upward[0]) / 2
    mesh_table = vd.grid_to_table(mesh)
    for j in range(A.shape[1]):
        A[:, j] = hm.prism_gravity(
            coordinates=coordenadas,
            prisms=[[
                mesh_table.horizontal[j] - size_horizontal,
                mesh_table.horizontal[j] + size_horizontal,
                -100e3, 100e3,
                mesh_table.upward[j] - size_vertical,
                mesh_table.upward[j] + size_vertical,
            ]],
            density=1,
            disable_checks=True,
            field="g_z",
        )

    # Constroi as matrizes de regularização
    # Pesos de profundidade
    weights = 1 / (coordenadas[2][0] - mesh_table.upward) ** (potencia / 2)
    # weights = np.sum(A**2, axis=0) ** (potencia / 4)
    W_depth = np.diag(weights / weights.max()) # / np.linalg.norm(dados)
    # Norma mínima
    W_damping = W_depth.T @ W_depth
    # Suavidade na horizontal e vertical
    R_h, R_v = finite_diff_matrices(mesh.density.shape)
    R_h = R_h @ W_depth
    R_v = R_v @ W_depth
    W_smooth_h = R_h.T @ R_h
    W_smooth_v = R_v.T @ R_v

    # Estima as densidades
    estimate = scipy.linalg.solve(
        A.T @ A + damping * W_damping + smoothness * W_smooth_h + smoothness * W_smooth_v,
        A.T @ dados,
    )
    mesh["density"].values = estimate.reshape(mesh.density.shape)

    # Calcula os dados preditos
    predicted = A @ estimate

    return mesh, predicted
  

def finite_diff_matrices(shape):
    "Produz matrizes de derivadas dos parametros para regularização"
    ny, nx = shape
    nderivs =  + (ny - 1) * nx
    I, J, V = [], [], []
    deriv = 0
    param = 0
    for i in range(ny):
        for j in range(nx - 1):
            I.extend([deriv, deriv])
            J.extend([param, param + 1])
            V.extend([1, -1])
            deriv += 1
            param += 1
        param += 1
    R_h = scipy.sparse.coo_matrix((V, (I, J)), ((nx - 1) * ny, nx * ny)).tocsr()
    I, J, V = [], [], []
    deriv = 0
    param = 0
    for i in range(ny - 1):
        for j in range(nx):
            I.extend([deriv, deriv])
            J.extend([param, param + nx])
            V.extend([1, -1])
            deriv += 1
            param += 1
    R_v = scipy.sparse.coo_matrix((V, (I, J)), ((ny - 1) * nx, nx * ny)).tocsr()
    return R_h, R_v
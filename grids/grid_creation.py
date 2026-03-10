import torch
import numpy as np
from scipy.spatial import ConvexHull
import spaudiopy as spa
import matplotlib.pyplot as plt

# Generate P points on the sphere
def generate_sphere_points(P, plot):
    """
    Generate points almost uniformly distributed on a sphere.

    Parameters:
    P (int): Number of points to generate.
    plot: flag to plot the points

    Returns:
    numpy.ndarray: Array of shape (num_points, 3) containing the Spherical coordinates of the points.
    """
    # Starting icosahedron
    vertices = icosahedron_vertices()
    faces = ConvexHull(vertices).simplices

    # Estimate how many subdivisions we need to get at least P points
    # Number of vertices after subdivision ~ (n_subdivisions^2 * initial_faces)
    n_faces = len(faces)
    n_subdivisions = round(
        np.sqrt(P / n_faces) + 0.5
    )  # num points grows as ~(n_subdivisions ** 2) per face (Asymptotic growth)

    # Subdivide the icosahedron
    for i in range(n_subdivisions):
        vertices, faces = subdivide(vertices, faces, i)
        ## Plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r', s=100, label='Vertices')
        # for simplex in faces:
        #     triangle = vertices[simplex]
        #     ax.add_collection3d(Poly3DCollection([triangle], color='cyan', edgecolor='k', linewidths=1, alpha=0.7))
        # plt.show()

    points = torch.tensor(vertices[:P])
    if plot:
        spa.plot.hull(spa.decoder.get_hull(*points.T))
        plt.title(f"{len(points)} Points on Sphere")
    return cart2sph(points)


# Define the 12 vertices of the icosahedron
def icosahedron_vertices():
    """
    Generate the vertices of an icosahedron.

    Returns:
    numpy.ndarray: Array of shape (12, 3) containing the Cartesian coordinates of the vertices.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]
    )
    # Normalize vertices to lie on the sphere
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return vertices


# Subdivide triangle face into smaller triangles
def subdivide(vertices, faces, n):
    """
    Subdivides each triangle face of the icosahedron into smaller triangles.

    Parameters:
    vertices - Vertices of the current mesh
    faces - Triangle faces as indices of vertices
    n - Number of subdivisions

    Returns:
    new_vertices - The vertices after subdivision
    new_faces - The faces (triangles) after subdivision
    """

    def midpoint(v1, v2):
        return (v1 + v2) / 2

    new_vertices = list(vertices)
    midpoint_cache = {}

    def add_vertex(v):
        # Normalize the vertex to project onto the sphere
        v = v / np.linalg.norm(v)
        new_vertices.append(v)
        return len(new_vertices) - 1

    def get_midpoint_index(v1_idx, v2_idx):
        smaller_idx = min(v1_idx, v2_idx)
        larger_idx = max(v1_idx, v2_idx)
        key = (smaller_idx, larger_idx)

        if key not in midpoint_cache:
            mid = midpoint(vertices[v1_idx], vertices[v2_idx])
            midpoint_cache[key] = add_vertex(mid)

        return midpoint_cache[key]

    new_faces = []
    for tri in faces:
        v0, v1, v2 = tri

        a = get_midpoint_index(v0, v1)
        b = get_midpoint_index(v1, v2)
        c = get_midpoint_index(v2, v0)

        new_faces.append([v0, a, c])
        new_faces.append([v1, b, a])
        new_faces.append([v2, c, b])
        new_faces.append([a, b, c])

    return np.stack(new_vertices), np.stack(new_faces)


def cart2sph(cart_coords):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    x (torch.ndarray): Array of x coordinates.
    y (torch.ndarray): Array of y coordinates.
    z (torch.ndarray): Array of z coordinates.

    Returns:
    torch.ndarray: Array of spherical coordinates (r, theta, phi).
    """
    x = cart_coords[:, 0]
    y = cart_coords[:, 1]
    z = cart_coords[:, 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.arccos(z / r)
    phi = torch.arctan2(y, x)
    return torch.stack((r, theta, phi),dim=1)

